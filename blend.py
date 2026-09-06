"""Stage 3a: choose ensemble weights and post-processing from evidence.

    python -u blend.py                 # pick weights, test post-processing
    python -u blend.py --objective late

Reads the tail predictions `tune.py` persisted and answers three questions
without fitting anything:

1. **Which members, at what weights?** Caruana greedy selection with
   replacement over rank-transformed predictions. Rank rather than probability
   because the members are on different scales and average precision reads only
   the ordering. Greedy rather than equal-weight because equal-weight is known
   to fail here: on the previous feature set XGBoost and CatBoost both scored
   below every LightGBM config, so averaging all families landed *below* the
   best single model. A member that hurts simply never gets picked.

2. **At which horizon?** Each candidate appears twice -- `@late` (trained to
   05-14, scoring 46-62 days ahead) and `@recent` (trained to 06-28, scoring
   1-17 days ahead) -- over the *same rows*. The real test window spans 1-62
   days past its cutoff, so it lies between the two. The default objective is
   the mean of both, which asks for a blend that works at either horizon rather
   than one tuned to whichever geometry happened to be luckier.

3. **Does entity post-processing help?** The IEEE-CIS winners replaced every
   prediction for a client with that client's mean. That works when the label
   is a property of the client; here fraud is per-transaction and a customer
   has many legitimate rows, so full replacement would destroy the within-
   customer ordering that AP is made of. The mild version -- shrink each score
   a little toward its entity's mean -- is testable, so it is tested rather
   than assumed.

Nothing here is adopted on a point estimate. The tail's measured 95% band is
+/-0.027 AP, so every reported gain is also run through the paired bootstrap
against the best single model.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config as C
from src import evaluate as E
from src import validation as V
from src.io_utils import get_stream

TAIL_PREDS = C.PROCESSED / "tune_tail_preds.parquet"
BLEND_SPEC = C.PROCESSED / "blend_spec_v2.json"
ROUND_BOOK = C.PROCESSED / "tune_rounds.json"
POST_OUT = C.PROCESSED / "postprocess_results.csv"

NOISE_FLOOR = 0.003


def load_preds() -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """-> {member: {"late": p, "recent": p}}, y."""
    if not TAIL_PREDS.exists():
        raise SystemExit("run tune.py first")
    vp = pd.read_parquet(TAIL_PREDS)
    y = vp.pop("_y").to_numpy()
    members: dict[str, dict[str, np.ndarray]] = {}
    for col in vp.columns:
        name, _, tag = col.partition("@")
        members.setdefault(name, {})[tag] = vp[col].to_numpy()
    # Only members present at both cutoffs can be judged across horizons.
    both = {k: v for k, v in members.items() if {"late", "recent"} <= set(v)}
    dropped = sorted(set(members) - set(both))
    if dropped:
        print(f"  ignoring {dropped} (missing one cutoff)")
    return both, y


def as_ranks(members) -> dict[str, dict[str, np.ndarray]]:
    n = len(next(iter(members.values()))["late"])
    return {k: {t: rankdata(p) / n for t, p in v.items()} for k, v in members.items()}


def objective_fn(y, which: str):
    if which == "late":
        return lambda c: V_ap(y, c["late"])
    if which == "recent":
        return lambda c: V_ap(y, c["recent"])
    return lambda c: 0.5 * (V_ap(y, c["late"]) + V_ap(y, c["recent"]))


def V_ap(y, p):
    from src.validation import ap

    return ap(y, p)


def hill_climb(ranks, y, obj, n_iter: int = 30):
    """Greedy selection with replacement, stopped at the peak.

    Members are added one at a time, repeats allowed so a strong model can
    accumulate weight. Iterating past the peak only adds noise, so the history
    is truncated to its argmax rather than run to completion.
    """
    names = list(ranks)
    chosen, cur, hist = [], None, []
    for _ in range(n_iter):
        best = (-1.0, None, None)
        for n in names:
            k = len(chosen)
            cand = {t: (ranks[n][t] if cur is None else (cur[t] * k + ranks[n][t]) / (k + 1))
                    for t in ("late", "recent")}
            a = obj(cand)
            if a > best[0]:
                best = (a, n, cand)
        chosen.append(best[1])
        cur = best[2]
        hist.append(best[0])
    k = int(np.argmax(hist)) + 1
    chosen = chosen[:k]
    weights = {n: chosen.count(n) / k for n in sorted(set(chosen))}
    blended = {t: sum(w * ranks[n][t] for n, w in weights.items())
               for t in ("late", "recent")}
    return weights, hist[k - 1], blended


# ---------------------------------------------------------------------------
# entity post-processing
# ---------------------------------------------------------------------------
def shrink_to_entity(p: np.ndarray, keys: np.ndarray, alpha: float,
                     stat: str = "mean") -> np.ndarray:
    """`(1-a)*score + a*(entity statistic)`, on ranks.

    a=0 is the identity and a=1 is the IEEE-CIS "replace with the client
    statistic" move. Everything in between trades within-entity resolution for
    between-entity evidence sharing.

    Four statistics, encoding different beliefs:

    * `mean` -- "this entity is broadly risky". Dilutes one strong signal
      across the entity's quiet rows.
    * `max` -- "one suspicious transaction condemns the rest". The fraud-ring /
      account-takeover story, and what the IEEE-CIS post-process was really
      exploiting.
    * `cummean` / `cummax` -- the same two, **past-only**: row `t` sees only the
      entity's rows at or before `t`. The plain forms look across the whole
      scored set, so an early test row would be adjusted using a row two months
      later. The rules permit non-target aggregation over test rows only "in a
      strictly-past-only way", and these are the forms that satisfy it. Assumes
      `p` and `keys` arrive in time order, which they do -- the stream is sorted
      by `(timestamp, transaction_id)` throughout.
    """
    s = pd.Series(p)
    g = s.groupby(pd.Series(keys), sort=False)
    stats = {
        "mean": lambda: g.transform("mean"),
        "max": lambda: g.transform("max"),
        "cummean": lambda: g.transform(lambda v: v.expanding().mean()),
        "cummax": lambda: g.cummax(),
    }
    return (1.0 - alpha) * p + alpha * stats[stat]().to_numpy()


def test_postprocessing(blended, y, df, tail_mask, obj) -> pd.DataFrame:
    ent = {
        "customer": df.loc[tail_mask, C.CUSTOMER].to_numpy(),
        "merchant": df.loc[tail_mask, C.MERCHANT].to_numpy(),
        "device": df.loc[tail_mask, C.DEVICE].to_numpy(),
    }
    base_late, base_recent = V_ap(y, blended["late"]), V_ap(y, blended["recent"])
    rows = [{"variant": "none", "stat": "-", "alpha": 0.0, "ap_late": base_late,
             "ap_recent": base_recent, "objective": obj(blended),
             "d_late": 0.0, "verdict": "reference"}]
    for name, keys in ent.items():
        for stat in ("mean", "max", "cummean", "cummax"):
            for a in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                cand = {t: shrink_to_entity(blended[t], keys, a, stat)
                        for t in ("late", "recent")}
                al, ar = V_ap(y, cand["late"]), V_ap(y, cand["recent"])
                d = al - base_late
                rows.append({"variant": name, "stat": stat, "alpha": a,
                             "ap_late": al, "ap_recent": ar,
                             "objective": obj(cand), "d_late": d,
                             "verdict": "no result (< noise floor)" if abs(d) < NOISE_FLOOR
                             else ("better" if d > 0 else "worse")})
    return pd.DataFrame(rows)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--objective", default="mean", choices=["mean", "late", "recent"])
    ap_.add_argument("--boot", type=int, default=400)
    args = ap_.parse_args()

    members, y = load_preds()
    ranks = as_ranks(members)
    obj = objective_fn(y, args.objective)
    br = float(y.mean())
    print(f"\n{len(members)} members, {len(y):,} tail rows, base rate {br:.4f}")

    print("\n=== SINGLES ===")
    singles = []
    for n, r in ranks.items():
        singles.append({"member": n, "ap_late": V_ap(y, r["late"]),
                        "ap_recent": V_ap(y, r["recent"]), "objective": obj(r)})
    st = pd.DataFrame(singles).sort_values("objective", ascending=False)
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best_single = st.iloc[0]["member"]

    print("\n=== BLENDS ===")

    def equal(names):
        w = {n: 1 / len(names) for n in names}
        return w, {t: np.mean([ranks[n][t] for n in names], axis=0)
                   for t in ("late", "recent")}

    # Three a-priori groupings, none of which learns a weight from the tail:
    #   core     the five hyperparameter configs -- what submission 2 shipped
    #   lgb_all  every LightGBM member, including the recency and window variants
    #   all      every member, including XGBoost and CatBoost
    # Membership comes from the round book so a member's family is read from
    # what was fitted, not guessed from its name.
    kinds = {}
    if ROUND_BOOK.exists():
        kinds = {k: v.get("kind", "lgb") for k, v in
                 json.loads(ROUND_BOOK.read_text()).items()}
    core = [n for n in ranks if n.startswith("lgb")]
    lgb_all = [n for n in ranks if kinds.get(n, "lgb") == "lgb"]
    weights, hc_obj, hc_blend = hill_climb(ranks, y, obj)

    options = {f"best_single ({best_single})": ({best_single: 1.0}, ranks[best_single])}
    for label, names in (("equal_core", core), ("equal_lgb_all", lgb_all),
                         ("equal_all", list(ranks))):
        if names and label not in options:
            options[label] = equal(names)
    options["hill_climb"] = (weights, hc_blend)
    has_far = "far" in next(iter(ranks.values()))
    for label, (ww_, b) in options.items():
        far = ""
        if has_far:
            # The blend's own far-horizon read: rebuilt from the same weights
            # rather than hill-climbed, because `tail_far` is a stress test and
            # selecting on it would defeat the point of having one.
            fb = sum(w2 * ranks[n2]["far"] for n2, w2 in ww_.items())
            far = f" far={V_ap(y, fb):.4f}"
        print(f"  {label:<28s} late={V_ap(y, b['late']):.4f} "
              f"recent={V_ap(y, b['recent']):.4f}{far} obj={obj(b):.4f}")

    # Adopt the *simplest* option, and upgrade only against evidence. Ordered
    # by how much the tail window is allowed to influence the answer:
    #
    #   equal_core     no weight is learned from the tail at all, and the
    #                  membership is the one already on the leaderboard
    #   equal_lgb_all  adds the recency/window variants: more averaging, but
    #                  members that individually measured as no-result
    #   equal_all      adds XGBoost and CatBoost -- naive cross-family averaging
    #                  is exactly what failed before, so it must earn its way
    #   hill_climb     every weight is fitted to 1,068 positives
    #
    # `equal_core` is the incumbent because averaging several fits of the same
    # family is variance reduction, which needs no justification from the tail,
    # and because it is what submission 2 already measured on the leaderboard --
    # so anything that replaces it replaces a known quantity.
    # A more selective option replaces it only if it clears the run-to-run floor
    # *and* the paired interval excludes zero. Picking the argmax of four point
    # estimates on this window is how `significant_blocks_only` happened.
    # Two classes of option, and they must not be compared on the same terms.
    #
    # **Unweighted** options (a single model, or an equal average over an
    # a-priori group) learn nothing from the tail. Nothing about them was
    # chosen by looking at these 1,068 positives, so their measured objective
    # is an honest estimate and the best one simply wins.
    #
    # **Fitted** options -- the hill-climb -- choose their weights on the very
    # rows the paired bootstrap then resamples. The bootstrap cannot see that
    # bias; it reports the interval of a quantity that was already optimised
    # against the data. This repo has the receipt: `significant_blocks_only`
    # cleared a paired test on `tail_late` at +0.0045 and went *negative* on
    # `tail_recent`. So a fitted blend must beat the best unweighted option by
    # **twice** the run-to-run floor before it is believed, and the hill-climb
    # here does not come close -- it clears `equal_core` by +0.0036 but the
    # best unweighted option by +0.0008, which is the part that matters.
    unweighted = [k for k in options if k != "hill_climb"]
    pick = max(unweighted, key=lambda k: obj(options[k][1]))
    print(f"  best unweighted: {pick} (obj {obj(options[pick][1]):.4f})")
    if "hill_climb" in options:
        d = E.paired_ap_delta(y, options[pick][1]["late"],
                              options["hill_climb"][1]["late"], n_boot=args.boot)
        gap = obj(options["hill_climb"][1]) - obj(options[pick][1])
        take = gap >= 2 * NOISE_FLOOR and d["lo"] > 0
        print(f"  hill_climb vs {pick}: objective {gap:+.4f}, tail_late "
              f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  -> "
              f"{'adopt' if take else 'reject (fitted weights need 2x the floor)'}")
        if take:
            pick = "hill_climb"

    w, blended = options[pick]
    print(f"\n  chosen: {pick}")
    for n, ww in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"    {n:<16s} {ww:.3f}")

    print("\n=== ENTITY POST-PROCESSING ===")
    df = get_stream()
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    tail_mask = V.tail_late_fold().val_mask(ts, is_test)
    post = test_postprocessing(blended, y, df, tail_mask, obj)
    print(post.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    post.to_csv(POST_OUT, index=False)

    # Adopted only if it clears the run-to-run floor AND the paired interval
    # excludes zero AND it does not go backwards at the other horizon. Sixteen
    # variants were tried; the best of sixteen beats the reference by chance
    # often enough that a bare point estimate would be worthless here.
    # Only the past-only forms are eligible for adoption. `max`/`mean` over the
    # whole scored set would adjust a 2026-07-16 test row using a 2026-09-15
    # one; the rules permit aggregation over test rows only "in a strictly
    # past-only way". They are still measured and printed, because the gap
    # between the two-sided and causal versions is itself the finding.
    CAUSAL = ("cummax", "cummean")
    eligible = post[post["stat"].isin(CAUSAL)]
    best_post = (eligible.sort_values("objective", ascending=False).iloc[0]
                 if len(eligible) else post.iloc[0])
    use_post = None
    if best_post["variant"] != "none" and best_post["d_late"] >= NOISE_FLOOR:
        keys = df.loc[tail_mask, {"customer": C.CUSTOMER, "merchant": C.MERCHANT,
                                  "device": C.DEVICE}[best_post["variant"]]].to_numpy()
        cand_late = shrink_to_entity(blended["late"], keys, float(best_post["alpha"]),
                                     best_post["stat"])
        pd_ = E.paired_ap_delta(y, blended["late"], cand_late, n_boot=args.boot)
        d_recent = best_post["ap_recent"] - V_ap(y, blended["recent"])
        # The alpha below it must also be positive. Entities pool ~2.6x harder
        # on the real test set than on the tail (customer rows/entity 2.9 -> 7.5,
        # device 4.6 -> 16.0), so an alpha sitting on a knife edge here would be
        # applied at a materially different effective strength there. Only a
        # variant whose whole lower shoulder is positive survives that mismatch.
        same = post[(post["variant"] == best_post["variant"]) &
                    (post["stat"] == best_post["stat"]) &
                    (post["alpha"] < best_post["alpha"])]
        shoulder = bool(len(same) == 0 or (same["d_late"] > 0).all())
        print(f"\n  best variant {best_post['variant']}/{best_post['stat']} "
              f"a={best_post['alpha']}: tail_late {pd_['delta']:+.4f} "
              f"[{pd_['lo']:+.4f}, {pd_['hi']:+.4f}], tail_recent {d_recent:+.4f}, "
              f"lower-alpha shoulder {'positive' if shoulder else 'NOT positive'}")
        if pd_["lo"] > 0 and d_recent > -NOISE_FLOOR and shoulder:
            use_post = {"entity": best_post["variant"], "stat": best_post["stat"],
                        "alpha": float(best_post["alpha"])}
            print(f"  adopting {use_post}")
    if use_post is None:
        print("\n  no post-processing adopted")

    far_ap = None
    if has_far:
        far_ap = V_ap(y, sum(ww * ranks[n]["far"] for n, ww in w.items()))
        print(f"\n  chosen blend at 75-91 days ahead (tail_far): {far_ap:.4f}")

    spec = {
        "objective": args.objective,
        "strategy": pick,
        "weights": w,
        "postprocess": use_post,
        "tail_late_ap": V_ap(y, blended["late"]),
        "tail_recent_ap": V_ap(y, blended["recent"]),
        "tail_far_ap": far_ap,
    }
    BLEND_SPEC.write_text(json.dumps(spec, indent=2))
    print(f"\nwrote {BLEND_SPEC}")


if __name__ == "__main__":
    main()
