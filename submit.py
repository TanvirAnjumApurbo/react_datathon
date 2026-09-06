"""Stage 3b: refit the chosen ensemble on all labelled data and write a submission.

    python -u submit.py --name s2_stage1_blend
    python -u submit.py --name s3_recency --weights '{"hl_14d": 1.0}'
    python -u submit.py --name s4_diverse --spec data/processed/blend_spec_v2.json

Each run writes `submissions/<name>.csv` plus `submissions/<name>.json`
recording exactly what produced it -- members, weights, seeds, round counts,
post-processing, and the local tail scores the choice was made on. Submissions
are capped at 10 for the whole competition and 5 per day, so "what was this
one spent to learn" has to survive longer than the terminal scrollback.

Round counts
------------
Members are early-stopped inside `tune.py` at the 2026-05-14 cutoff (492k
training rows) and refit here on all 732k labelled rows. Rounds are scaled by
`sqrt(n_final / n_ref)` = 1.22x, the same damped rule `harness.run_folds` uses
for a solo cutoff: more data supports a longer fit, but not proportionally, and
on a drifting target overshooting is the more expensive mistake. (The previous
hard-coded 1.2x turns out to be the same number, now with a reason.)

Seeds
-----
LightGBM here runs multi-threaded and not `deterministic=True`, so two runs of
identical code differ by ~0.003 AP. Averaging ranks over several seeds removes
most of that from the submission itself -- worth doing when the whole decision
budget is 10 uploads.

Member cache
------------
Each member's seed-averaged test ranking is cached under
`data/processed/test_preds/`, keyed on the member, the seed count, the
determinism switch and the feature-source hash. Composing a second blend from
members already fitted is then a weighted sum rather than another twenty
minutes of boosting, which is what makes it affordable to answer four different
questions in one sitting instead of one.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config as C
from src import harness as H
from src import validation as V
from src.io_utils import get_stream
from src.features import encoding
from tune import FITTERS, recency_weights, train_mask

ROUND_BOOK = C.PROCESSED / "tune_rounds.json"
BLEND_SPEC = C.PROCESSED / "blend_spec_v2.json"
SUB_DIR = C.ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)
CACHE_DIR = C.PROCESSED / "test_preds"
CACHE_DIR.mkdir(exist_ok=True)


def entity_shrink(p: np.ndarray, keys: np.ndarray, alpha: float,
                  stat: str = "mean") -> np.ndarray:
    """Mirror of `blend.shrink_to_entity`, applied to the test predictions.

    Uses only model outputs and an entity id -- no labels, no fitting -- so it
    is post-processing, not a statistic learned on test.
    """
    g = pd.Series(p).groupby(pd.Series(keys), sort=False).transform(stat).to_numpy()
    return (1.0 - alpha) * p + alpha * g


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--name", required=True, help="submission name (no extension)")
    ap_.add_argument("--spec", default=None, help="blend spec json (default: blend_spec_v2)")
    ap_.add_argument("--weights", default=None, help='inline JSON, e.g. \'{"lgb_base": 1.0}\'')
    ap_.add_argument("--seeds", type=int, default=3)
    ap_.add_argument("--no-post", action="store_true", help="ignore any post-processing in the spec")
    ap_.add_argument("--no-gnn", action="store_true")
    ap_.add_argument("--deterministic", action="store_true",
                     help="bit-reproducible LightGBM fits (~2x slower). Rulebook "
                          "8.2 forfeits a position that cannot be reproduced from "
                          "the submitted notebook, so use this for anything that "
                          "might be selected for private scoring.")
    ap_.add_argument("--note", default="", help="what this submission is spent to learn")
    args = ap_.parse_args()

    book = json.loads(ROUND_BOOK.read_text())
    if args.weights:
        weights, post, spec_src, local = json.loads(args.weights), None, "inline", {}
    else:
        # `--spec` may be a full path or a bare filename inside data/processed.
        if args.spec:
            path = pathlib.Path(args.spec)
            if not path.exists():
                path = C.PROCESSED / args.spec
        else:
            path = BLEND_SPEC
        if not path.exists():
            raise SystemExit(f"no --weights and no blend spec at {path}; run blend.py first")
        spec = json.loads(path.read_text())
        weights = spec["weights"]
        post = None if args.no_post else spec.get("postprocess")
        spec_src = str(path)
        local = {k: spec.get(k) for k in ("tail_late_ap", "tail_recent_ap", "strategy")}

    missing = [m for m in weights if m not in book]
    if missing:
        raise SystemExit(f"members not in {ROUND_BOOK}: {missing}")

    df = get_stream()
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)

    print("loading base features...")
    base = H.load_base(df, use_gnn=not args.no_gnn)
    te = encoding.build(df, fit_cutoff=C.TRAIN_END)   # production encoder
    X = pd.concat([base, te], axis=1)
    print(f"  matrix {X.shape[1]} cols")

    Xs = X[is_test]
    n_test = int(is_test.sum())
    acc, total_w, members = np.zeros(n_test), 0.0, []

    for name, w_cfg in sorted(weights.items(), key=lambda kv: -kv[1]):
        cfg = book[name]
        # A member's seed-averaged test ranking depends on the member, the seed
        # count, the determinism switch and the feature matrix -- not on which
        # blend is asking for it. Caching on exactly those means the second,
        # third and fourth submission variants cost seconds instead of a refit
        # each, which is the difference between exploring four ideas today and
        # exploring one.
        key = f"{name}__s{args.seeds}{'__det' if args.deterministic else ''}" \
              f"__{H.feature_source_hash()}"
        cache = CACHE_DIR / f"{key}.npy"
        if cache.exists():
            member_rank = np.load(cache)
            members.extend(f"{name}_s{s}" for s in range(args.seeds))
            print(f"  {name:<14s} w={w_cfg:.3f} (cached)")
        else:
            tm = train_mask(ts, is_test, C.TRAIN_END, cfg["train_days"])
            n_final = int(tm.sum())
            rounds = max(int(cfg["rounds"] * (n_final / max(cfg["n_train"], 1)) ** 0.5), 50)
            w_row = recency_weights(ts, tm, C.TRAIN_END, cfg["halflife"])
            Xt, yt = X[tm], y[tm]

            seed_ranks = []
            for s in range(args.seeds):
                t = time.time()
                params = dict(cfg["params"])
                if args.deterministic and cfg["kind"] == "lgb":
                    params.update(deterministic=True, force_row_wise=True, num_threads=8)
                _, _, pred_fn = FITTERS[cfg["kind"]](
                    Xt, yt, w_row, None, None, params, rounds, C.SEED + 100 * s)
                p = pred_fn(Xs)
                seed_ranks.append(rankdata(p) / n_test)
                members.append(f"{name}_s{s}")
                print(f"  {name}_s{s:<2d} w={w_cfg:.3f} rounds={rounds:5d} "
                      f"train={n_final:,} {time.time() - t:5.0f}s", flush=True)
            # Average seeds first, then apply the member's blend weight: the
            # seeds are the same model, the members are not.
            member_rank = np.mean(seed_ranks, axis=0)
            np.save(cache, member_rank)
        acc += w_cfg * member_rank
        total_w += w_cfg

    final = acc / total_w

    if post:
        keys = df.loc[is_test, {"customer": C.CUSTOMER, "merchant": C.MERCHANT,
                                "device": C.DEVICE}[post["entity"]]].to_numpy()
        stat = post.get("stat", "mean")
        final = entity_shrink(final, keys, float(post["alpha"]), stat)
        print(f"  post-processed: shrink {post['alpha']} toward {post['entity']} {stat}")

    # AP is invariant to any monotone rescale; [0,1] just keeps the file
    # readable as probabilities and satisfies the submission format.
    final = (final - final.min()) / (final.max() - final.min())

    sub = pd.DataFrame({C.ID_COL: df.loc[is_test, C.ID_COL].to_numpy(), C.TARGET: final})
    sample = pd.read_csv(C.SAMPLE_SUB)
    sub = sample[[C.ID_COL]].merge(sub, on=C.ID_COL, how="left")
    assert sub[C.TARGET].notna().all(), "missing predictions for some test ids"
    assert len(sub) == len(sample), "submission row count mismatch"
    assert sub[C.TARGET].between(0, 1).all(), "predictions outside [0,1]"

    out = SUB_DIR / f"{args.name}.csv"
    sub.to_csv(out, index=False)
    (SUB_DIR / f"{args.name}.json").write_text(json.dumps({
        "name": args.name, "note": args.note, "spec": spec_src,
        "weights": weights, "seeds": args.seeds, "postprocess": post,
        "members": members, "local": local,
        "n_features": int(X.shape[1]),
        "te_feedback_delay_d": C.TE_FEEDBACK_DELAY_D,
        "use_signup_inconsistency": C.USE_SIGNUP_INCONSISTENCY,
    }, indent=2, default=str))

    print(f"\nwrote {out}  ({len(sub):,} rows, {len(members)} models)")
    print(f"  mean={final.mean():.4f} p99={np.quantile(final, 0.99):.4f}")

    # If an earlier submission exists, say how different this one actually is.
    # Two files that rank the test set almost identically will score almost
    # identically, and spending an upload to find that out is the mistake this
    # check exists to prevent.
    prev = sorted(p for p in SUB_DIR.glob("*.csv") if p != out)
    for p in prev[-3:]:
        o = pd.read_csv(p)
        j = sub.merge(o, on=C.ID_COL, suffixes=("", "_o"))
        rho = np.corrcoef(rankdata(j[C.TARGET]), rankdata(j[f"{C.TARGET}_o"]))[0, 1]
        k = max(int(0.01 * len(j)), 1)
        top_a = set(j.nlargest(k, C.TARGET)[C.ID_COL])
        top_b = set(j.nlargest(k, f"{C.TARGET}_o")[C.ID_COL])
        print(f"  vs {p.name:<28s} rank rho={rho:.4f}  top-1% overlap={len(top_a & top_b) / k:.3f}")


if __name__ == "__main__":
    main()
