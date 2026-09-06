"""Scoring and diagnosis. One place, so every stage reports the same numbers.

Submission 1 taught this repo that a single averaged AP is not a usable read on
this dataset. Local `primary_62d` said 0.7252; the leaderboard said 0.52708.
Nothing was broken -- the fold averaged a regime change away. Every helper here
exists to make that specific mistake hard to repeat:

* `fold_scores` scores one prediction vector on *several* windows at once, so
  the tail is never optional. `tail_late` shares `primary_62d`'s cutoff, so
  reporting both costs one extra masked AP call, not a second model fit.
* `weekly_ap` and `horizon_ap` break a fold open. If the weekly series has a
  cliff in it, the fold mean is a fiction and you should be reading the last
  rows, not the average.
* Every AP is reported next to **lift over that slice's own base rate**. Weekly
  base rates here range 0.0139-0.0193, and raw AP moves with them, so two raw
  APs from different windows are not directly comparable.
* `ap_ci` and `paired_ap_delta` put a number on the noise. The tail window has
  ~67k rows and ~1,050 positives; its sampling band is wide. A point estimate
  from it is not evidence on its own, and the paired test is the right way to
  ask "is B better than A", because both are scored on the identical rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import validation as V


# ---------------------------------------------------------------------------
# point scores
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Score:
    """One AP measurement, always carrying the context needed to read it."""

    slice: str
    n: int
    n_pos: int
    base_rate: float
    ap: float
    lift: float

    def __str__(self) -> str:
        return (
            f"{self.slice:<14s} AP={self.ap:.4f} ({self.lift:5.1f}x base) "
            f"n={self.n:,} pos={self.n_pos:,}"
        )


def score(y: np.ndarray, p: np.ndarray, slice_name: str = "") -> Score:
    """AP plus the base rate it has to be read against."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or np.nansum(y) == 0:
        return Score(slice_name, len(y), 0, float("nan"), float("nan"), float("nan"))
    br = float(np.nanmean(y))
    a = V.ap(y, p)
    return Score(slice_name, len(y), int(np.nansum(y)), br, a, a / br)


def scores_frame(scores: list[Score]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in scores])


# ---------------------------------------------------------------------------
# multi-window scoring of a single prediction vector
# ---------------------------------------------------------------------------
def fold_scores(
    ts: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    is_test: np.ndarray,
    folds: list[V.Fold] | None = None,
) -> pd.DataFrame:
    """Score one full-stream prediction vector on every fold's val window.

    `p` must be aligned to the full stream (same length and order as `ts`).
    Use NaN for rows the model did not predict; those rows are dropped from the
    window they fall in, and a window with no predictions is skipped.

    Folds that share a cutoff can be scored from a single fit. That is the
    whole point: `primary_62d` and `tail_late` are the same model read two
    ways, and the second read is the one that has tracked the leaderboard.
    """
    folds = folds or V.selection_folds()
    p = np.asarray(p, dtype=float)
    out = []
    for f in folds:
        m = f.val_mask(ts, is_test) & np.isfinite(p)
        if m.sum() == 0:
            continue
        out.append(score(y[m], p[m], f.name))
    return scores_frame(out)


def weekly_ap(
    ts: pd.Series, y: np.ndarray, p: np.ndarray, freq: str = "W"
) -> pd.DataFrame:
    """AP per calendar week. The check that a fold mean is not hiding a cliff.

    Weeks with fewer than 30 positives are reported but should be ignored: AP
    on a handful of positives is mostly noise.
    """
    p = np.asarray(p, dtype=float)
    m = np.isfinite(p)
    g = pd.Series(pd.to_datetime(ts)[m]).dt.to_period(freq).dt.end_time
    rows = []
    for wk, idx in pd.Series(np.arange(int(m.sum()))).groupby(g.to_numpy()):
        i = idx.to_numpy()
        s = score(y[m][i], p[m][i], str(pd.Timestamp(wk).date()))
        rows.append({**asdict(s), "reliable": s.n_pos >= 30})
    return pd.DataFrame(rows).rename(columns={"slice": "week_ending"})


def horizon_ap(
    ts: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    cutoff: pd.Timestamp,
    edges_d: tuple[int, ...] = (0, 7, 15, 31, 46, 62),
) -> pd.DataFrame:
    """AP by days-after-cutoff. Shows decay with forecast horizon.

    On this dataset the decay is not gradual: it was flat for six weeks and
    then fell off, because the fraud generator changed rather than because the
    forecast got stale. `weekly_ap` separates those two readings -- horizon
    decay and calendar drift are confounded inside a single fold.
    """
    p = np.asarray(p, dtype=float)
    m = np.isfinite(p)
    days = (pd.to_datetime(ts)[m] - pd.Timestamp(cutoff)).dt.total_seconds().to_numpy() / 86_400.0
    rows = []
    for lo, hi in zip(edges_d[:-1], edges_d[1:]):
        sel = (days >= lo) & (days < hi)
        if sel.sum() == 0:
            continue
        rows.append({**asdict(score(y[m][sel], p[m][sel], f"{lo}-{hi}d")), "lo_d": lo})
    rows.append({**asdict(score(y[m], p[m], "all")), "lo_d": 9_999})
    return pd.DataFrame(rows).rename(columns={"slice": "horizon"})


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def ap_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = C.SEED,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for AP on one slice.

    The tail window carries ~1,050 positives. Average precision on that many
    positives has a real sampling band, and quoting a bare fourth decimal from
    it invites reading noise as progress. Resampling is over rows, which
    captures the dominant term (which positives landed in the window).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        boots[b] = V.ap(yb, p[idx]) if yb.sum() > 0 else np.nan
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return V.ap(y, p), float(lo), float(hi)


def paired_ap_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 400,
    seed: int = C.SEED,
) -> dict:
    """Is B better than A on these rows? Paired bootstrap on AP(B) - AP(A).

    Paired, because both candidates are scored on identical rows: the shared
    sampling noise cancels and what is left is the difference that matters.
    Comparing two independent confidence intervals instead would be far more
    conservative than the question deserves.

    Returns the observed delta, its interval, and `p_better` -- the fraction of
    resamples where B wins. Treat |delta| below the ~0.003 run-to-run floor as
    no result regardless of what the interval says.
    """
    y = np.asarray(y, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            deltas[b] = np.nan
            continue
        deltas[b] = V.ap(yb, p_b[idx]) - V.ap(yb, p_a[idx])
    lo, hi = np.nanquantile(deltas, [0.025, 0.975])
    return {
        "ap_a": V.ap(y, p_a),
        "ap_b": V.ap(y, p_b),
        "delta": V.ap(y, p_b) - V.ap(y, p_a),
        "lo": float(lo),
        "hi": float(hi),
        "p_better": float(np.nanmean(deltas > 0)),
    }


# ---------------------------------------------------------------------------
# per-feature drift diagnosis
# ---------------------------------------------------------------------------
def feature_ap_early_late(
    X: pd.DataFrame,
    y: np.ndarray,
    ts: pd.Series,
    mask: np.ndarray,
    split: pd.Timestamp = C.TAIL_START,
    min_ap: float = 0.02,
) -> pd.DataFrame:
    """Standalone AP of every feature, early vs late, and the retained fraction.

    This is the table that explained submission 1: `amt_ratio_median` kept 0.64
    of its power across the split while `cust_dt` kept 0.89 and `gnn_cd_cos`
    0.99. A feature whose retention is low is one the test period will not pay
    for, however good its overall gain looks.

    Scored inside `mask` only -- pass a validation mask, never the rows the
    model trained on.
    """
    ts = pd.to_datetime(ts)
    early = mask & (ts < split).to_numpy()
    late = mask & (ts >= split).to_numpy()
    rows = []
    for c in X.columns:
        # An unordered categorical has no meaningful ranking, so "AP of this
        # column used directly as a score" is not defined for it. Coercing it
        # would produce all-NaN and report the base rate, which reads as a real
        # (and uniformly identical) number -- worse than saying nothing.
        if not pd.api.types.is_numeric_dtype(X[c]) or isinstance(
            X[c].dtype, pd.CategoricalDtype
        ):
            rows.append(
                {"feature": c, "ap_early": np.nan, "ap_late": np.nan,
                 "retained": np.nan, "sign": np.nan, "kind": "categorical"}
            )
            continue
        v = X[c].to_numpy(dtype=float)
        a_e, sign = V.single_feature_ap(v[early], y[early])
        a_l = V.ap(y[late], sign * np.where(np.isfinite(v[late]), v[late], -1e9))
        rows.append(
            {
                "feature": c,
                "ap_early": a_e,
                "ap_late": a_l,
                "retained": a_l / a_e if a_e > 0 else np.nan,
                "sign": sign,
                "kind": "numeric",
            }
        )
    out = pd.DataFrame(rows)
    out["material"] = out.ap_early >= min_ap
    return out.sort_values("ap_early", ascending=False, na_position="last").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def print_scores(frame: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_fold_report(
    ts: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    is_test: np.ndarray,
    cutoff: pd.Timestamp,
    folds: list[V.Fold] | None = None,
    label: str = "",
) -> pd.DataFrame:
    """The standard read on a candidate: windows, then weeks, then horizon."""
    fs = fold_scores(ts, y, p, is_test, folds)
    print_scores(fs, f"WINDOW SCORES {label}".strip())

    finite = np.isfinite(np.asarray(p, dtype=float))
    val = finite & (~is_test) & (pd.to_datetime(ts) > pd.Timestamp(cutoff)).to_numpy()
    if val.sum():
        wk = weekly_ap(ts[val], y[val], np.asarray(p, dtype=float)[val])
        print_scores(
            wk[["week_ending", "n", "n_pos", "base_rate", "ap", "lift", "reliable"]],
            "WEEKLY (the fold mean is only honest if this is flat)",
        )
        print_scores(
            horizon_ap(ts[val], y[val], np.asarray(p, dtype=float)[val], cutoff),
            "BY FORECAST HORIZON",
        )
    return fs
