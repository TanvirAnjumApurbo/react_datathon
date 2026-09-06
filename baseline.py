"""The reference number. Everything else is measured against this.

    .venv/Scripts/python.exe -u baseline.py

A bare LightGBM on the **raw columns only** -- amount, account age, hour, day of
week and the five low-cardinality categoricals. No history, no aggregation, no
encoding. This is what the dataset gives you for free, and the whole premise of
the repo is that leakage-safe behavioural features beat it by a wide margin.
Until that margin is measured on the tail window, it is an assumption.

Three things this answers that nothing else in the repo does:

1. **How much of the score is feature engineering?** 201 engineered features
   scored 0.7252 on `primary_62d`. Against what? A raw-column model is the
   only honest denominator.
2. **Does the drift hit the raw columns or the engineered ones?** Both models
   are scored on `primary_62d` and on `tail_late`. If the raw baseline holds
   its tail score better than the engineered pipeline does, the engineering is
   buying old-regime performance and paying for it in the test period -- which
   is exactly the failure submission 1 ran into.
3. **Does `scale_pos_weight` do anything for average precision?** The research
   report recommends it over resampling, which is well supported. Whether it
   helps a *ranking* metric at all is a separate question, and cheap to settle:
   three weightings, same features, same folds.

Writes `data/processed/baseline_results.csv`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C
from src import evaluate as E
from src import harness as H
from src import validation as V
from src.io_utils import get_stream

OUT = C.PROCESSED / "baseline_results.csv"

# The raw columns, and nothing else. `account_age_days` is included because it
# ships with the data; note it is monotone in calendar time for any fixed
# account, which is exactly the drift trap `entity.py` pairs with an `*_frac`
# version -- here it stays raw, because the point is to measure what raw gives.
RAW_NUM = ["amount_bdt", "account_age_days"]
RAW_CAT = C.LOW_CARD_CATS


def raw_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns plus the two calendar fields any model would derive."""
    X = pd.DataFrame(index=df.index)
    X["amount_bdt"] = df["amount_bdt"].astype(np.float32)
    X["account_age_days"] = df["account_age_days"].astype(np.float32)
    X["hour"] = df[C.TIME_COL].dt.hour.astype(np.int8)
    X["dayofweek"] = df[C.TIME_COL].dt.dayofweek.astype(np.int8)
    for c in RAW_CAT:
        X[c] = df[c].astype("category")
    return X


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--boot", type=int, default=400,
                     help="bootstrap resamples for the tail confidence band")
    args = ap_.parse_args()

    df = get_stream()
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)

    X = raw_matrix(df)
    print(f"raw matrix: {X.shape[0]:,} x {X.shape[1]}  ({list(X.columns)})")
    print(V.describe_folds(df).to_string(index=False))

    n_neg, n_pos = int((y[~is_test] == 0).sum()), int((y[~is_test] == 1).sum())
    spw = n_neg / n_pos
    print(f"\nclass balance: {n_pos:,} pos / {n_neg:,} neg  -> scale_pos_weight={spw:.1f}")

    # Three imbalance treatments over identical features and folds. The metric
    # only reads the ordering, so a global reweighting of the positive class
    # can only help through its effect on tree structure and leaf values -- it
    # is not obvious a priori that it does anything at all here.
    variants = {
        "raw_plain": {},
        "raw_spw": {"scale_pos_weight": spw},
        "raw_spw_half": {"scale_pos_weight": spw / 2},
    }

    all_res = []
    fits_by_variant = {}
    for name, extra in variants.items():
        print(f"\n--- {name} ---", flush=True)
        res, fits = H.run_folds(
            lambda _cutoff: X,
            y, ts, is_test,
            folds=V.all_folds(),
            params=extra,
            label=name,
            want_gain=True,
        )
        all_res.append(res)
        fits_by_variant[name] = fits

    res = pd.concat(all_res, ignore_index=True)
    print("\n=== BASELINE: raw columns only ===")
    piv = res.pivot_table(index="candidate", columns="slice", values="ap")
    order = [c for c in ("tail_late", "tail_recent", "primary_62d", "wf_mar", "wf_apr", "wf_may")
             if c in piv.columns]
    print(piv[order].to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nlift over each slice's own base rate:")
    pivl = res.pivot_table(index="candidate", columns="slice", values="lift")
    print(pivl[order].to_string(float_format=lambda x: f"{x:.1f}"))

    # The tail window is small. Put a band on it so nobody reads its fourth
    # decimal as a result.
    best = res[res.slice == "tail_late"].sort_values("ap", ascending=False).iloc[0]
    fits = fits_by_variant[best["candidate"]]
    tail = V.tail_late_fold()
    m = tail.val_mask(ts, is_test)
    p = fits[tail.train_end].pred
    pt, lo, hi = E.ap_ci(y[m], p[m], n_boot=args.boot)
    print(
        f"\ntail_late for {best['candidate']}: AP={pt:.4f} "
        f"[{lo:.4f}, {hi:.4f}] 95% bootstrap over {args.boot} resamples"
    )
    print(f"  -> the tail window's own sampling band is +/-{(hi - lo) / 2:.4f} AP. "
          f"Differences smaller than that are not results.")

    # Does the raw model show the same weekly cliff the engineered one did? If
    # the collapse is visible with nothing but amount, hour and account age,
    # then it is the data that moved and not something the feature pipeline
    # introduced.
    E.print_fold_report(
        ts, y, p, is_test, cutoff=tail.train_end, folds=V.selection_folds(),
        label=f"({best['candidate']})",
    )

    # Where the raw signal sits, and how much of it survives into the tail.
    pf = V.primary_fold()
    vm = pf.val_mask(ts, is_test)
    fa = E.feature_ap_early_late(X, y, ts, vm, split=C.TAIL_START, min_ap=0.0)
    print("\n=== RAW FEATURES: standalone AP, early vs late in primary_62d ===")
    print("(late = the tail window; `retained` below ~0.7 means the test period "
          "will not pay for it)")
    print(fa.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    gain = fits_by_variant["raw_plain"][pf.train_end].gain
    if gain is not None:
        print("\n=== RAW FEATURES: LightGBM gain share (primary cutoff) ===")
        print((100 * gain.sort_values(ascending=False)).to_string(
            float_format=lambda x: f"{x:.1f}%"))

    res.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
