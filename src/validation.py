"""Time-based validation. This is what protects us from the leaderboard.

The competition forbids random K-fold outright:

    "random K-fold cross-validation is not appropriate for this task, because
     it lets information from a customer's future transactions leak into the
     validation score for their past transactions"

The real task geometry is: train on everything up to a cutoff, then score a
**62-day forward block** starting the very next day (test runs 2026-07-16 to
2026-09-15, immediately after the 2026-07-15 train cutoff, with no gap).

`PRIMARY_FOLD` reproduces that geometry exactly. It is **no longer the
selection target**: submission 1 scored 0.7252 on it and 0.52708 on the public
leaderboard, because its 62-day window averages six weeks of one fraud regime
together with two weeks of the next.

Selection now happens on the **tail folds** (`tail_late`, `tail_recent`), which
score only 2026-06-29 -> 07-15 -- the newest labelled regime, the one the test
period sits inside. `tail_late` shares `primary_62d`'s cutoff, so it is a free
second read on the same trained model; it scored 0.516 against an LB of 0.527
and is the only slice that has tracked reality. `tail_recent` moves the cutoff
up to the day before the window, which is the only geometry that can see
recency adaptation working.

The secondary walk-forward folds exist to measure *stability* across time --
given the organisers' explicit drift warning and the 60/40 public/private
split, a feature set that scores slightly lower but varies less across folds is
the better bet.

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


def tail_late_fold() -> Fold:
    """The LB-tracking slice: 46-62 days ahead, newest regime.

    Same cutoff as `primary_62d`, so a model already fitted for the primary
    fold can be scored on this one at zero extra training cost -- just a
    different validation mask. Use `Fold.val_mask` on the same predictions.
    """
    return Fold(**C.TAIL_LATE_FOLD)


def tail_recent_fold() -> Fold:
    """Same rows as `tail_late`, but trained right up to the day before.

    Short horizon, matched regime. This is the fold that can answer "does
    adapting to recent data help", which `primary_62d` structurally cannot.
    """
    return Fold(**C.TAIL_RECENT_FOLD)


def tail_far_fold() -> Fold:
    """Same rows again, 75-91 days ahead. A stress read, never a selection target.

    Deliberately one step beyond the real task's 62-day maximum. Its use is as
    a tie-breaker between configs that `tail_late` cannot separate: the far half
    of the test window is the part `tail_late` measures least well, and a config
    that degrades gracefully here is the safer bet for it.
    """
    return Fold(**C.TAIL_FAR_FOLD)


def far_wide_fold() -> Fold:
    """The 04-16 -> 06-28 window at the `tail_far` cutoff, for stopping only."""
    return Fold(**C.FAR_WIDE_FOLD)


def es_late_fold() -> Fold:
    """The stopping window for the 2026-05-14 cutoff: 05-15 -> 06-28.

    Deliberately ends the day before the tail begins. `primary_62d` cannot be
    used for this even though it is wider, because `tail_late` is a *subset* of
    it -- 1,068 of its 4,028 positives -- so stopping there would pick the round
    count using the labels of the window being reported.
    """
    return Fold(**C.ES_LATE_FOLD)


#: Cutoffs that have a dedicated stopping window, built to end the day before
#: `TAIL_START` so it cannot overlap any tail window.
_ES_BY_CUTOFF = {
    C.ES_LATE_FOLD["train_end"]: es_late_fold,
    C.FAR_WIDE_FOLD["train_end"]: far_wide_fold,
}


def stopping_fold(folds: list[Fold]) -> Fold:
    """Which window to early-stop on, given the windows about to be reported.

    Uses the registered window for this cutoff where one exists. Those windows
    are constructed to be disjoint from every *tail* window -- the ones
    selection actually happens on -- which is the property that matters. They
    do overlap `primary_62d`, unavoidably: no window at the 05-14 cutoff can
    stop a 62-day fold without touching it. That is accepted because
    `primary_62d` is reported for continuity and is not selected on.

    Falls back to the widest reported window where nothing is registered (the
    walk-forward folds, each of which is alone at its cutoff and is a stability
    read rather than a decision input).
    """
    fn = _ES_BY_CUTOFF.get(folds[0].train_end)
    if fn is not None:
        es = fn()
        assert all(es.val_end < f.val_start or es.val_start > f.val_end
                   for f in folds if f.name.startswith("tail_")), \
            f"stopping window {es.name} overlaps a tail window"
        return es
    return max(folds, key=lambda f: f.horizon_days)


def secondary_folds() -> list[Fold]:
    return [Fold(**f) for f in C.SECONDARY_FOLDS]


def selection_folds() -> list[Fold]:
    """The folds a decision may be made on. Believe the tail when they differ."""
    return [tail_late_fold(), primary_fold()]


def all_folds() -> list[Fold]:
    return [primary_fold(), tail_late_fold(), tail_recent_fold(), *secondary_folds()]


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
