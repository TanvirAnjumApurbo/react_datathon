"""Feature-validation harness. NOT model development.

Runs a single fixed LightGBM over the walk-forward folds to answer three
questions and nothing else:

  1. does the split behave (train AP >> val AP >> base rate, with no fold
     scoring implausibly well)?
  2. which feature blocks actually carry weight?
  3. is the target-encoding block earning its leakage risk?

Hyperparameter search, ensembling and calibration are deliberately out of
scope -- they belong to the modelling pass, not the preprocessing pass.

Target encoding is rebuilt **per fold** with that fold's cutoff. Reusing the
production encoder (fitted to 2026-07-15) inside a fold ending 2026-05-14 would
leak validation-period labels into the encoder and inflate the score.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C
from src import validation as V
from src.io_utils import get_stream
from src.features import (
    amount, behaviour, encoding, entity, graph, temporal, velocity,
)

PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=64,
    min_data_in_leaf=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=0,
    verbosity=-1,
    seed=C.SEED,
)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--no-gnn", action="store_true")
    ap_.add_argument("--rounds", type=int, default=1500)
    args = ap_.parse_args()

    import lightgbm as lgb

    df = get_stream()
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)

    print("building label-free blocks (shared across folds)...", flush=True)
    parts, blocks = [], {}
    for name, fn in [
        ("temporal", temporal.build),
        ("amount", amount.build),
        ("velocity", velocity.build),
        ("entity", entity.build),
        ("behaviour", behaviour.build),
        ("graph", lambda d: graph.build(d, use_gnn=not args.no_gnn, verbose=False)),
    ]:
        blk = fn(df)
        blocks[name] = list(blk.columns)
        parts.append(blk)
    base = pd.concat(parts, axis=1)
    print(f"  {base.shape[1]} label-free features")
    print(V.describe_folds(df).to_string(index=False))

    results, gains = [], {}
    for fold in V.all_folds():
        # Rebuild the encoder against THIS fold's cutoff.
        te = encoding.build(df, fit_cutoff=fold.train_end)
        X = pd.concat([base, te], axis=1)

        tm, vm = fold.train_mask(ts, is_test), fold.val_mask(ts, is_test)
        dtr = lgb.Dataset(X[tm], label=y[tm])
        dva = lgb.Dataset(X[vm], label=y[vm], reference=dtr)
        model = lgb.train(
            PARAMS,
            dtr,
            num_boost_round=args.rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pv = model.predict(X[vm], num_iteration=model.best_iteration)
        pt = model.predict(X[tm], num_iteration=model.best_iteration)
        val_ap, tr_ap = V.ap(y[vm], pv), V.ap(y[tm], pt)
        base_rate = float(y[vm].mean())
        results.append(
            {
                "fold": fold.name,
                "horizon_d": fold.horizon_days,
                "n_val": int(vm.sum()),
                "base_rate": base_rate,
                "val_ap": val_ap,
                "x_base": val_ap / base_rate,
                "train_ap": tr_ap,
                "best_iter": model.best_iteration,
            }
        )
        print(
            f"  {fold.name:12s} val_AP={val_ap:.4f} ({val_ap/base_rate:5.1f}x base) "
            f"train_AP={tr_ap:.4f} iters={model.best_iteration}",
            flush=True,
        )
        g = pd.Series(model.feature_importance("gain"), index=X.columns)
        gains[fold.name] = g / g.sum()

    res = pd.DataFrame(results)
    print("\n=== FOLD SUMMARY ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nval_AP mean={res.val_ap.mean():.4f}  std={res.val_ap.std():.4f}  "
          f"(stability matters more than peak -- the private LB is a different 40%)")

    G = pd.DataFrame(gains)
    G["mean_gain"] = G.mean(axis=1)
    G = G.sort_values("mean_gain", ascending=False)
    print("\n=== TOP 30 FEATURES BY GAIN ===")
    print((100 * G[["mean_gain"]].head(30)).to_string(float_format=lambda x: f"{x:.2f}%"))

    blocks["encoding"] = [c for c in G.index if c.startswith("te_")]
    print("\n=== GAIN BY BLOCK ===")
    for b, cols in blocks.items():
        cols = [c for c in cols if c in G.index]
        print(f"  {b:9s} {100*G.loc[cols,'mean_gain'].sum():5.1f}%  ({len(cols)} features)")

    res.to_csv(C.PROCESSED / "fold_results.csv", index=False)
    G.to_csv(C.PROCESSED / "feature_gain.csv")
    print(f"\nwrote {C.PROCESSED / 'fold_results.csv'} and feature_gain.csv")


if __name__ == "__main__":
    main()
