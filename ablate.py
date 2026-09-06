"""Does this change help on the window that has tracked the leaderboard?

    .venv/Scripts/python.exe -u ablate.py --set blocks     # leave-one-block-out
    .venv/Scripts/python.exe -u ablate.py --set drift      # the counter question
    .venv/Scripts/python.exe -u ablate.py --set stage1     # new feature blocks
    .venv/Scripts/python.exe -u ablate.py --set all

Every candidate is fitted at the `primary_62d` cutoff and read twice: once over
the full 62 days, once over the 17-day tail. One fit, two windows -- and the
tail is the one that agreed with the leaderboard.

Point estimates on the tail are not enough to act on. It carries ~1,050
positives and its bootstrap band is about +/-0.027 AP, so almost any single
comparison "fits inside the noise" if you only look at two numbers. Two things
fix that:

* **A paired bootstrap against the reference.** Both candidates score the
  identical rows, so the shared sampling noise cancels and the interval is on
  the *difference*, which is far tighter than either interval alone.
* **Agreement across windows.** A change that helps the tail and the primary
  fold and the walk-forward folds is a different kind of evidence from one
  that helps only the window it was selected on.

The verdict column combines them, and it is deliberately conservative: a
candidate has to clear the ~0.003 run-to-run floor *and* win the paired test
before it reads as anything other than "no result".
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src import evaluate as E
from src import harness as H
from src import validation as V
from src.features import encoding
from src.io_utils import get_stream

OUT = C.PROCESSED / "ablation_results.csv"
ADV = C.PROCESSED / "adversarial_features.csv"

# The run-to-run floor. LightGBM runs with num_threads=0 and without
# deterministic=True, so two fits of identical code on identical data differ;
# measured top-500 overlap is 98.4%. Anything under this is not a result.
NOISE_FLOOR = 0.003


# ---------------------------------------------------------------------------
# candidate sets
# ---------------------------------------------------------------------------
def adversarial_artifacts() -> list[str]:
    """Features the drift diagnostic flagged as unbounded time counters."""
    if not ADV.exists():
        return []
    adv = pd.read_csv(ADV)
    adv = adv[adv.comparison.str.startswith("ENGINEERED")]
    if "position_artifact" not in adv.columns:
        return []
    return adv.loc[adv.position_artifact, "feature"].tolist()


def severe_artifacts(auc: float = 0.90) -> list[str]:
    """Only the ones whose test values sit essentially outside the train range."""
    if not ADV.exists():
        return []
    adv = pd.read_csv(ADV)
    adv = adv[adv.comparison.str.startswith("ENGINEERED")]
    return adv.loc[adv.position_artifact & (adv.auc >= auc), "feature"].tolist()


# Columns added in this pass. Grouped so the ablation can ask what the whole
# Stage 1 feature push bought, not only what each piece did.
STAGE1_AMOUNT = {
    "amt_rank_in_cust", "amt_round_500", "amt_cents", "amt_has_cents",
    "amt_magnitude",
    # Derived from the window list rather than matched by prefix: a prefix like
    # "amt_ratio_mean_1" would also have to avoid catching
    # "amt_ratio_mean_by_categ", and that is the kind of near-miss that quietly
    # ablates the wrong columns.
    *(f"cust_amt_mean_{w}" for w in C.RFM_WINDOWS_S),
    *(f"amt_ratio_mean_{w}" for w in C.RFM_WINDOWS_S),
    *(f"amt_share_{w}" for w in C.RFM_WINDOWS_S),
}


def stage1_amount_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c in STAGE1_AMOUNT]


class Candidate:
    """A feature set plus the encoder settings it is scored with."""

    def __init__(self, keep: list[str], te_delay: float | None = None):
        self.keep = list(keep)
        self.te_delay = te_delay          # None -> config default

    def __len__(self) -> int:
        return len(self.keep)


def build_candidates(which: str, cols: list[str], prov: dict[str, str]) -> dict:
    """name -> Candidate. The reference must be first."""
    cand: dict[str, Candidate] = {"reference": Candidate(cols)}
    blocks = sorted(set(prov.values()))

    if which in ("blocks", "all"):
        for b in blocks:
            keep = [c for c in cols if prov.get(c) != b]
            if keep and len(keep) < len(cols):
                cand[f"drop_block:{b}"] = Candidate(keep)

    if which in ("drift", "all"):
        art = set(adversarial_artifacts())
        sev = set(severe_artifacts())
        if sev:
            cand[f"drop_severe_counters({len(sev)})"] = Candidate(
                [c for c in cols if c not in sev])
        if art:
            cand[f"drop_all_counters({len(art)})"] = Candidate(
                [c for c in cols if c not in art])
        # The amount family is the most drift-exposed block by measurement:
        # amt_ratio_median kept 0.64 of its standalone power across the split.
        # Worth asking directly whether leaning on it costs the tail.
        amt = [c for c in cols if prov.get(c) == "amount"]
        if amt:
            cand[f"drop_amount_block({len(amt)})"] = Candidate(
                [c for c in cols if c not in set(amt)])

    if which in ("stage1", "all"):
        # What did this pass add? Drop it all and see.
        new_amt = set(stage1_amount_cols(cols))
        beh = {c for c in cols if prov.get(c, "").startswith("behaviour")}
        added = new_amt | beh
        if added:
            cand[f"drop_all_stage1({len(added)})"] = Candidate(
                [c for c in cols if c not in added])
        if new_amt:
            cand[f"drop_stage1_amount({len(new_amt)})"] = Candidate(
                [c for c in cols if c not in new_amt])
        # Sub-blocks of the behaviour module, so habit / repetition / travel /
        # youth get separate verdicts. Their standalone reads on the tail
        # already differ by an order of magnitude, so one lumped answer would
        # hide more than it settled.
        for b in sorted({prov.get(c, "") for c in beh}):
            drop = {c for c in cols if prov.get(c) == b}
            if drop:
                cand[f"drop_{b}({len(drop)})"] = Candidate(
                    [c for c in cols if c not in drop])

        # The feedback delay, at the handbook's 7 days and at nothing. Same
        # features, different encoder -- the only fair way to ask whether
        # delaying the training rows' view of the labels pays.
        for d in (0.0, 30.0):
            cand[f"te_delay_{d:g}d"] = Candidate(cols, te_delay=d)

    if which == "delay":
        # The encoder delay on its own, so it can be confirmed on windows it
        # was not selected on. Run with `--folds all`.
        for d in (0.0, 3.0, 14.0, 30.0):
            cand[f"te_delay_{d:g}d"] = Candidate(cols, te_delay=d)

    if which in ("trim", "all"):
        # The two largest positive deltas from the block sweep were dropping
        # the graph block (+0.0020) and dropping the new amount columns
        # (+0.0022). Neither clears the noise floor alone. Testing them
        # together is the cheap way to find out whether that is two halves of
        # one real effect or two draws from the same noise -- and it is a
        # question worth settling, because 43 columns is a lot of surface to
        # carry for nothing.
        drop_graph = {c for c in cols if prov.get(c) == "graph"}
        drop_amt = set(stage1_amount_cols(cols))
        if drop_graph and drop_amt:
            both = drop_graph | drop_amt
            cand[f"drop_graph_and_stage1_amount({len(both)})"] = Candidate(
                [c for c in cols if c not in both])
        # And the leanest set that kept every block with a significant verdict.
        keep_only = {c for c in cols
                     if prov.get(c) in ("amount", "velocity", "entity", "temporal")
                     or prov.get(c) == "encoding"}
        keep_only -= drop_amt
        if keep_only and len(keep_only) < len(cols):
            cand[f"significant_blocks_only({len(keep_only)})"] = Candidate(
                sorted(keep_only))

    return cand


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(
    df: pd.DataFrame,
    base: pd.DataFrame,
    candidates: dict[str, list[str]],
    folds: list[V.Fold],
    n_boot: int,
) -> pd.DataFrame:
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)

    # One encoder per (cutoff, delay), rebuilt against that fold. Cached across
    # candidates because the encoder does not depend on which columns are kept
    # -- only on the cutoff and the feedback delay.
    te_cache: dict[tuple, pd.DataFrame] = {}

    def te_for(cutoff, delay):
        key = (cutoff, delay)
        if key not in te_cache:
            te_cache[key] = encoding.build(df, fit_cutoff=cutoff, delay_d=delay)
        return te_cache[key]

    tail = V.tail_late_fold()
    tail_m = tail.val_mask(ts, is_test)

    rows, tail_preds = [], {}
    for name, cand in candidates.items():
        t0 = time.time()
        keep_set = set(cand.keep)
        delay = cand.te_delay

        def build_X(cutoff, keep_set=keep_set, delay=delay):
            # The encoder is rebuilt per cutoff, so its columns are joined here
            # rather than living in `base`. They are still ordinary members of
            # the keep set -- that is what lets `drop_block:encoding` mean what
            # it says instead of silently keeping the encoder.
            X = pd.concat([base, te_for(cutoff, delay)], axis=1)
            return X[[c for c in X.columns if c in keep_set]]

        res, fits = H.run_folds(
            build_X, y, ts, is_test, folds=folds, label=name, verbose=False
        )
        pred = fits[tail.train_end].pred
        tail_preds[name] = pred[tail_m]

        row = {"candidate": name, "n_features": len(cand),
               "te_delay_d": delay if delay is not None else C.TE_FEEDBACK_DELAY_D,
               "secs": round(time.time() - t0)}
        for _, r in res.iterrows():
            row[f"ap_{r['slice']}"] = r["ap"]
            row[f"lift_{r['slice']}"] = r["lift"]
        rows.append(row)
        print(
            f"  {name:<32s} tail={row.get('ap_tail_late', float('nan')):.4f} "
            f"primary={row.get('ap_primary_62d', float('nan')):.4f} "
            f"({len(cand):3d} feat, {row['secs']:4d}s)",
            flush=True,
        )

    out = pd.DataFrame(rows)

    # Paired bootstrap on the tail, every candidate against the reference.
    ref = list(candidates)[0]
    y_tail = y[tail_m]
    deltas = []
    for name in candidates:
        if name == ref:
            deltas.append({"candidate": name, "tail_delta": 0.0, "tail_lo": 0.0,
                           "tail_hi": 0.0, "p_better": 0.5, "verdict": "reference"})
            continue
        d = E.paired_ap_delta(y_tail, tail_preds[ref], tail_preds[name], n_boot=n_boot)
        # Conservative by design. The interval alone would call some changes
        # significant that are smaller than two runs of identical code differ
        # by, so the floor is applied first.
        if abs(d["delta"]) < NOISE_FLOOR:
            verdict = "no result (< noise floor)"
        elif d["lo"] > 0:
            verdict = "BETTER"
        elif d["hi"] < 0:
            verdict = "WORSE"
        else:
            verdict = f"unclear (p={d['p_better']:.2f})"
        deltas.append({"candidate": name, "tail_delta": d["delta"], "tail_lo": d["lo"],
                       "tail_hi": d["hi"], "p_better": d["p_better"], "verdict": verdict})

    return out.merge(pd.DataFrame(deltas), on="candidate")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--set", dest="which", default="all",
                     choices=["blocks", "drift", "stage1", "trim", "delay", "all"])
    ap_.add_argument("--boot", type=int, default=400)
    ap_.add_argument("--no-gnn", action="store_true")
    ap_.add_argument("--folds", default="selection",
                     choices=["selection", "all"],
                     help="'selection' = tail_late + primary_62d from one fit")
    args = ap_.parse_args()

    df = get_stream()
    print("loading base features...")
    base = H.load_base(df, use_gnn=not args.no_gnn)
    prov = dict(H.base_provenance())
    if not prov:
        print("  (no provenance stamp; block ablations unavailable this run)")

    # Target-encoded columns are rebuilt per fold, so they are not in `base`.
    # Register their names anyway so the encoding block can be ablated like
    # any other -- `te_merchant_id` has been an untested open item for a while
    # precisely because nothing could drop it and remeasure.
    te_cols = list(encoding.build(df, fit_cutoff=V.primary_fold().train_end).columns)
    for c in te_cols:
        prov[c] = "encoding"

    folds = V.selection_folds() if args.folds == "selection" else V.all_folds()
    universe = list(base.columns) + te_cols
    cand = build_candidates(args.which, universe, prov)
    print(f"\n{len(cand)} candidates over {len(universe)} features "
          f"({base.shape[1]} label-free + {len(te_cols)} encoded), "
          f"folds: {[f.name for f in folds]}\n")

    res = run(df, base, cand, folds, args.boot)

    print("\n=== ABLATION: change vs reference, judged on the tail window ===")
    show = ["candidate", "n_features", "te_delay_d", "ap_tail_late",
            "ap_primary_62d", "tail_delta", "tail_lo", "tail_hi", "verdict"]
    show = [c for c in show if c in res.columns]
    print(res[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  tail_delta is AP(candidate) - AP(reference) on the same rows, with a "
          f"95% paired-bootstrap interval.\n  Anything inside +/-{NOISE_FLOOR} is "
          f"below the run-to-run floor and reads as no result.")

    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
