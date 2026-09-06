"""Vectorised strictly-past window aggregations.

Every "historical" feature in this project routes through here, so the
past-only guarantee is enforced in one place instead of being re-argued in
each feature module.

Definition of "past": the stream is totally ordered by (timestamp,
transaction_id). A row's history is every row of the same key that appears
*earlier in that order*. Rows sharing a timestamp are ordered by
transaction_id, which is itself assigned in arrival order -- this is what a
real-time scoring system would see, and it never looks forward.

Implementation
--------------
The stream is globally time-sorted, so within any group the timestamps are
sorted too. We build a composite key

    composite = group_code * BIG + ts_epoch      (BIG > max ts_epoch)

which is monotonically increasing once rows are sorted by (group, time).
A single global `np.searchsorted` then resolves the window start for every
row at once, with no Python-level loop over groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ts_epoch maxes out around 1.79e9; 2**31 comfortably exceeds it while keeping
# group_code * BIG inside int64 for the ~40k groups we have.
_BIG = np.int64(2**31)


class WindowIndex:
    """Reusable index over one grouping key, for many window sizes."""

    def __init__(self, group_code: np.ndarray, ts_epoch: np.ndarray):
        group_code = np.asarray(group_code, dtype=np.int64)
        ts = np.asarray(ts_epoch, dtype=np.int64)
        assert group_code.min() >= 0, "group codes must be non-negative"
        assert ts.max() < _BIG, "ts_epoch exceeds composite-key capacity"

        self.n = len(ts)
        # Stable sort by group; within a group the original (time) order holds.
        self.order = np.argsort(group_code, kind="stable")
        self.inv = np.empty(self.n, dtype=np.int64)
        self.inv[self.order] = np.arange(self.n, dtype=np.int64)

        self.g_sorted = group_code[self.order]
        self.t_sorted = ts[self.order]
        self.composite = self.g_sorted * _BIG + self.t_sorted
        assert np.all(np.diff(self.composite) >= 0), "composite key not monotone"

        # Position of each row within its own group (0 = first ever).
        starts = np.searchsorted(self.g_sorted, self.g_sorted, side="left")
        self.pos_in_group = np.arange(self.n, dtype=np.int64) - starts
        self.group_start = starts

    # -- primitives -------------------------------------------------------
    def _lo(self, window_s: int) -> np.ndarray:
        """Sorted-space index of the first row inside the trailing window."""
        target = self.g_sorted * _BIG + (self.t_sorted - np.int64(window_s))
        return np.searchsorted(self.composite, target, side="left")

    def prior_count(self) -> np.ndarray:
        """Number of strictly-earlier rows sharing this key (lifetime)."""
        return self.pos_in_group[self.inv]

    def count_in_window(self, window_s: int) -> np.ndarray:
        """Count of strictly-earlier rows within `window_s` seconds."""
        lo = self._lo(window_s)
        cnt = np.arange(self.n, dtype=np.int64) - lo
        return cnt[self.inv]

    def sum_in_window(self, values: np.ndarray, window_s: int) -> np.ndarray:
        """Sum of `values` over strictly-earlier rows within the window."""
        v = np.asarray(values, dtype=np.float64)[self.order]
        prefix = np.concatenate([[0.0], np.cumsum(v)])
        lo = self._lo(window_s)
        idx = np.arange(self.n, dtype=np.int64)
        out = prefix[idx] - prefix[lo]
        return out[self.inv]

    def prev_gap(self, lag: int = 1) -> np.ndarray:
        """Seconds since this key's `lag`-th previous occurrence (NaN if none)."""
        prev = np.full(self.n, np.nan)
        valid = self.pos_in_group >= lag
        idx = np.arange(self.n, dtype=np.int64)
        prev[valid] = (self.t_sorted[idx[valid]] - self.t_sorted[idx[valid] - lag]).astype(float)
        return prev[self.inv]

    def expanding_prev(self, values: np.ndarray, how: str) -> np.ndarray:
        """Expanding statistic over strictly-earlier rows of the key.

        `how` in {mean, std, max, min, median-ish}. Uses shift(1) semantics:
        the first row of any key gets NaN.
        """
        v = pd.Series(np.asarray(values, dtype=np.float64)[self.order])
        g = pd.Series(self.g_sorted)
        shifted = v.groupby(g).shift(1)
        exp = shifted.groupby(g).expanding()
        out = getattr(exp, how)().reset_index(level=0, drop=True).to_numpy()
        return out[self.inv]

    def prev_value(self, values: np.ndarray, lag: int = 1) -> np.ndarray:
        """The key's `lag`-th previous value of `values` (NaN if none)."""
        v = np.asarray(values, dtype=np.float64)[self.order]
        out = np.full(self.n, np.nan)
        valid = self.pos_in_group >= lag
        idx = np.arange(self.n, dtype=np.int64)
        out[valid] = v[idx[valid] - lag]
        return out[self.inv]

    def rolling_mean_last_k(self, values: np.ndarray, k: int) -> np.ndarray:
        """Mean of the key's previous `k` values (NaN until k exist)."""
        v = pd.Series(np.asarray(values, dtype=np.float64)[self.order])
        g = pd.Series(self.g_sorted)
        out = v.groupby(g).shift(1).groupby(g).rolling(k, min_periods=1).mean()
        out = out.reset_index(level=0, drop=True).to_numpy()
        return out[self.inv]


def first_occurrence_flag(*code_arrays: np.ndarray) -> np.ndarray:
    """1 where this row is the first-ever appearance of the key combination.

    Uses `cumcount() == 0` over the time-sorted stream, so it is past-only by
    construction.
    """
    keys = pd.MultiIndex.from_arrays([np.asarray(a) for a in code_arrays])
    df = pd.DataFrame({"_k": keys})
    return (df.groupby("_k", sort=False).cumcount() == 0).to_numpy().astype(np.int8)


def running_nunique(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Distinct `inner` values seen for each `outer` key *before* this row.

    Exact lifetime distinct count. Equivalent to a running set size, but
    computed as a cumulative sum of first-occurrence flags, which is O(n log n)
    instead of a Python set per row.
    """
    first = first_occurrence_flag(outer, inner).astype(np.int64)
    s = pd.Series(first)
    # cumsum includes the current row, so subtract it to stay strictly-past.
    return (s.groupby(pd.Series(np.asarray(outer)), sort=False).cumsum() - s).to_numpy()


def new_pairs_in_window(
    outer: np.ndarray, inner: np.ndarray, ts_epoch: np.ndarray, window_s: int
) -> np.ndarray:
    """How many *new* `inner` entities the `outer` key acquired in the window.

    Exact, and a sharper ramp-up signal than a windowed distinct count: a
    device that picked up 9 new customers in 24h is a farm spinning up, whereas
    a windowed distinct count would also fire for 9 regulars returning.
    """
    first = first_occurrence_flag(outer, inner).astype(np.float64)
    idx = WindowIndex(outer, ts_epoch)
    return idx.sum_in_window(first, window_s)
