"""Time-based validation. This is what protects us from the leaderboard.

The competition forbids random K-fold outright:

    "random K-fold cross-validation is not appropriate for this task, because
     it lets information from a customer's future transactions leak into the
     validation score for their past transactions"

The real task geometry is: train on everything up to a cutoff, then score a
**62-day forward block** starting the very next day (test runs 2026-07-16 to
2026-09-15, immediately after the 2026-07-15 train cutoff, with no gap).

`PRIMARY_FOLD` reproduces that geometry exactly and is the only fold used for
model selection. The secondary walk-forward folds exist to measure *stability*
across time -- given the organisers' explicit drift warning and the 60/40
public/private split, a feature set that scores slightly lower but varies less
across folds is the better bet.

Purging: every feature in this pipeline is strictly past-only, so a training
row never sees validation data, and a validation row using train-period history
is exactly what happens at test time. Heavy purging would therefore *mis*model
the task. The one exception is target encoding, whose expanding statistic can
blend across the boundary -- `embargo_mask` handles that case only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as C


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def train_mask(self, ts: pd.Series, is_test: np.ndarray) -> np.ndarray:
        return (~is_test) & (ts <= self.train_end).to_numpy()

    def val_mask(self, ts: pd.Series, is_test: np.ndarray) -> np.ndarray:
        return (
            (~is_test)
            & (ts >= self.val_start).to_numpy()
            & (ts <= self.val_end).to_numpy()
        )

    @property
    def horizon_days(self) -> int:
        return int((self.val_end - self.val_start).days) + 1


def primary_fold() -> Fold:
    return Fold(**C.PRIMARY_FOLD)


def secondary_folds() -> list[Fold]:
    return [Fold(**f) for f in C.SECONDARY_FOLDS]


def all_folds() -> list[Fold]:
    return [primary_fold(), *secondary_folds()]


def embargo_mask(ts: pd.Series, fold: Fold, seconds: int = C.EMBARGO_S) -> np.ndarray:
    """Rows in the `seconds` immediately before the cutoff.

    Only needed where an expanding target-encoding statistic could straddle the
    boundary; ordinary past-only features need no embargo.
    """
    lo = fold.train_end - pd.Timedelta(seconds=seconds)
    return ((ts > lo) & (ts <= fold.train_end)).to_numpy()


def ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The competition metric: sklearn.metrics.average_precision_score."""
    return float(average_precision_score(y_true, y_score))


def single_feature_ap(
    values: np.ndarray, y: np.ndarray, fill: float = -1e9
) -> tuple[float, float]:
    """AP of one raw feature used directly as a ranker, both polarities.

    Returns (best_ap, sign). Missing values sort to the bottom. Useful for
    catching a feature that is individually near-perfect, which is almost
    always leakage rather than a discovery.
    """
    v = np.asarray(values, dtype=np.float64)
    v = np.where(np.isfinite(v), v, fill)
    pos = ap(y, v)
    neg = ap(y, -v)
    return (pos, 1.0) if pos >= neg else (neg, -1.0)


def describe_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Row/positive counts per fold, for a sanity read before training."""
    ts = df[C.TIME_COL]
    is_test = df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy()
    rows = []
    for f in all_folds():
        tm, vm = f.train_mask(ts, is_test), f.val_mask(ts, is_test)
        rows.append(
            {
                "fold": f.name,
                "horizon_d": f.horizon_days,
                "n_train": int(tm.sum()),
                "n_val": int(vm.sum()),
                "train_pos": int(np.nansum(y[tm])),
                "val_pos": int(np.nansum(y[vm])),
                "val_base_rate": float(np.nanmean(y[vm])),
            }
        )
    return pd.DataFrame(rows)
