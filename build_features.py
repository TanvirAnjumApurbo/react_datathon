"""Build the full leakage-safe feature matrix.

    python build_features.py                 # everything, with the GNN tier
    python build_features.py --no-gnn        # skip Tier C (much faster)
    python build_features.py --check         # + run the truncation leakage test

Outputs
-------
data/processed/train_features.parquet
data/processed/test_features.parquet
data/processed/feature_manifest.json

Target encoding is written with `fit_cutoff = TRAIN_END`, which is the correct
setting for scoring the real test set. Backtests must rebuild that block with
their own fold cutoff -- see train_check.py -- otherwise validation rows read an
encoder that real test rows will never have.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src import leakage_checks as LC
from src.io_utils import get_stream
from src.features import (
    amount, behaviour, encoding, entity, graph, temporal, velocity,
)

BLOCKS = {
    "temporal": temporal.build,
    "amount": amount.build,
    "velocity": velocity.build,
    "entity": entity.build,
    "behaviour": behaviour.build,
    "encoding": encoding.build,
}


def build_matrix(df: pd.DataFrame, use_gnn: bool = True, verbose: bool = True) -> pd.DataFrame:
    parts, provenance = [], {}
    for name, fn in BLOCKS.items():
        t = time.time()
        block = fn(df)
        parts.append(block)
        sub = block.attrs.get("subblock", {})
        for c in block.columns:
            provenance[c] = f"{name}:{sub[c]}" if c in sub else name
        if verbose:
            print(f"  {name:9s} {block.shape[1]:3d} cols  {time.time()-t:6.1f}s", flush=True)

    t = time.time()
    g = graph.build(df, use_gnn=use_gnn, verbose=False)
    parts.append(g)
    for c in g.columns:
        provenance[c] = "graph"
    if verbose:
        print(f"  {'graph':9s} {g.shape[1]:3d} cols  {time.time()-t:6.1f}s", flush=True)

    F = pd.concat(parts, axis=1)
    F.attrs["provenance"] = provenance
    return F


def build_matrix_no_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Everything except target encoding and the graph.

    Used by the truncation leakage test: it must recompute on a truncated
    stream, and the label-free blocks are the ones whose past-only property we
    want proved row-by-row. The encoder is audited separately via its explicit
    `fit_cutoff`, and the graph via its snapshot construction.
    """
    return pd.concat(
        [
            temporal.build(df),
            amount.build(df),
            velocity.build(df),
            entity.build(df),
            behaviour.build(df),
        ],
        axis=1,
    )


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--no-gnn", action="store_true", help="skip the Tier C GNN block")
    ap_.add_argument("--check", action="store_true", help="run the truncation leakage test")
    args = ap_.parse_args()

    t0 = time.time()
    print("loading stream...", flush=True)
    df = get_stream()
    print(f"  {len(df):,} rows ({(~df.is_test).sum():,} train / {df.is_test.sum():,} test)")

    print("building features...", flush=True)
    F = build_matrix(df, use_gnn=not args.no_gnn)
    print(f"feature matrix: {F.shape[0]:,} x {F.shape[1]}")

    print("\nrunning leakage checks...", flush=True)
    LC.check_no_banned_columns(F)
    print("  banned columns .............. PASS")

    y = df[C.TARGET].to_numpy(dtype=float)
    ap_rep = LC.check_no_label_in_features(F, y)
    print(f"  no near-perfect feature ..... PASS (max single-feature AP {ap_rep.ap.max():.4f})")

    boundary = LC.check_boundary_continuity(F, df)
    n_flag = int((boundary.flag == "REVIEW").sum())
    print(f"  train/test boundary ......... {n_flag} flagged for review")

    tail = LC.check_tail_coverage(F, df)
    print(f"  end-of-stream coverage ...... PASS (max NaN excess {tail.excess.max():.4f})")

    if args.check:
        print("  truncation invariance ....... running (~60s)", flush=True)
        LC.check_truncation_invariance(build_matrix_no_labels, df, pd.Timestamp("2026-05-01"))
        print("  truncation invariance ....... PASS")

    is_test = df["is_test"].to_numpy()
    keys = df[[C.ID_COL, C.TIME_COL]].reset_index(drop=True)

    train_out = pd.concat([keys[~is_test], F[~is_test], df.loc[~is_test, [C.TARGET]]], axis=1)
    test_out = pd.concat([keys[is_test], F[is_test]], axis=1)
    train_out.to_parquet(C.PROCESSED / "train_features.parquet", index=False)
    test_out.to_parquet(C.PROCESSED / "test_features.parquet", index=False)

    prov = F.attrs["provenance"]
    ap_map = dict(zip(ap_rep.feature, ap_rep.ap))
    manifest = {
        "n_features": int(F.shape[1]),
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "te_fit_cutoff": str(C.TRAIN_END),
        "te_feedback_delay_d": C.TE_FEEDBACK_DELAY_D,
        "graph_snapshot_freq": C.GRAPH_SNAPSHOT_FREQ,
        "gnn_enabled": not args.no_gnn,
        "features": [
            {"name": c, "block": prov.get(c, "?"), "single_feature_ap_train": ap_map.get(c)}
            for c in F.columns
        ],
        "boundary_review": boundary[boundary.flag == "REVIEW"].feature.tolist(),
    }
    (C.PROCESSED / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote train_features.parquet {train_out.shape}")
    print(f"wrote test_features.parquet  {test_out.shape}")
    print(f"wrote feature_manifest.json  ({F.shape[1]} features)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
