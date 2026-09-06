"""Leakage guards. Run before every submission; fail loudly.

The competition treats temporal leakage as a rule violation "whether accidental
or deliberate", and top-15 teams have their pipeline reproduced. These checks
are the evidence that the feature set is past-only.

The centrepiece is `check_truncation_invariance`. If a feature for a row at
time t depends on anything at t' > t, then deleting all rows after a cutoff T
must change that feature for some row before T. Recomputing the whole pipeline
on a truncated stream and diffing is therefore a direct, assumption-free test
of the past-only property -- much stronger than eyeballing the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class LeakageError(AssertionError):
    """Raised when a check fails. Never catch this."""


def check_no_banned_columns(features: pd.DataFrame) -> None:
    """(1)+(2) No target, no id, no absolute time in the model matrix."""
    bad = sorted(set(features.columns) & C.BANNED_FEATURES)
    if bad:
        raise LeakageError(f"banned columns present in feature matrix: {bad}")

    suspicious = [
        c
        for c in features.columns
        if any(t in c.lower() for t in ("fraud", "target", "label", "_y", "epoch"))
    ]
    if suspicious:
        raise LeakageError(f"columns with target/time-like names: {suspicious}")


def check_first_row_is_null(
    features: pd.DataFrame, df: pd.DataFrame, key: str, cols: list[str]
) -> None:
    """(5) Every history statistic must be null on an entity's first ever row.

    A `shift(1)`-based expanding statistic has nothing to average on row 0. If
    a value appears there, the statistic included the current row.
    """
    first = df.groupby(f"{key}_code", sort=False).cumcount().to_numpy() == 0
    offenders = []
    for c in cols:
        if c not in features.columns:
            continue
        vals = features.loc[first, c]
        if vals.notna().any():
            offenders.append((c, int(vals.notna().sum())))
    if offenders:
        raise LeakageError(
            f"history features non-null on first-ever {key} row: {offenders}"
        )


def check_truncation_invariance(
    build_fn,
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    atol: float = 1e-5,
    sample_rows: int = 50_000,
) -> pd.DataFrame:
    """(3) The decisive test: features must not move when the future is deleted.

    Recomputes the entire feature pipeline on `df` truncated at `cutoff`, then
    compares against the full-stream features on the rows they share. Any
    column that differs is reading forward in time.

    `build_fn` must include every derivation, including anything the loader
    would otherwise bake in before this test runs. A customer-level summary
    computed inside `get_stream` once escaped this check precisely because it
    happened upstream of `build_fn`; anything derived per customer belongs in a
    feature module, not the loader.
    """
    full = build_fn(df)
    keep = (df[C.TIME_COL] <= cutoff).to_numpy()
    trunc = build_fn(df.loc[keep].reset_index(drop=True))

    n = int(keep.sum())
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(n, size=min(sample_rows, n), replace=False)

    rows = []
    for col in full.columns:
        if col not in trunc.columns:
            rows.append({"feature": col, "status": "MISSING_IN_TRUNCATED", "n_diff": -1})
            continue
        a = pd.to_numeric(full[col].to_numpy()[:n][idx], errors="coerce").astype(float)
        b = pd.to_numeric(trunc[col].to_numpy()[idx], errors="coerce").astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, atol=atol, rtol=1e-4) | both_nan
        n_diff = int((~close).sum())
        rows.append(
            {
                "feature": col,
                "status": "OK" if n_diff == 0 else "LEAKS_FUTURE",
                "n_diff": n_diff,
            }
        )

    report = pd.DataFrame(rows)
    leaks = report[report.status != "OK"]
    if len(leaks):
        raise LeakageError(
            "features change when the future is removed:\n"
            + leaks.to_string(index=False)
        )
    return report


def check_boundary_continuity(
    features: pd.DataFrame, df: pd.DataFrame, days: int = 14, max_shift: float = 0.35
) -> pd.DataFrame:
    """(4) Feature distributions must not jump at the train/test boundary.

    A discontinuity almost always means the feature was computed differently on
    each side -- typically a groupby fitted on train only. Reported as a
    normalised median shift; `amount_bdt` is the one known genuine exception
    (the fraud amount archetype really does thin out in the test period).
    """
    ts = df[C.TIME_COL]
    pre = ((ts > C.TRAIN_END - pd.Timedelta(days=days)) & (ts <= C.TRAIN_END)).to_numpy()
    post = ((ts >= C.TEST_START) & (ts < C.TEST_START + pd.Timedelta(days=days))).to_numpy()

    rows = []
    for col in features.columns:
        v = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
        a, b = v[pre], v[post]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 100 or len(b) < 100:
            continue
        ma, mb = np.median(a), np.median(b)
        scale = np.std(np.concatenate([a, b]))
        shift = abs(mb - ma) / scale if scale > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "median_pre": ma,
                "median_post": mb,
                "norm_shift": shift,
                "nan_pre": float(np.mean(~np.isfinite(v[pre]))),
                "nan_post": float(np.mean(~np.isfinite(v[post]))),
            }
        )
    report = pd.DataFrame(rows).sort_values("norm_shift", ascending=False)
    report["flag"] = np.where(report.norm_shift > max_shift, "REVIEW", "ok")
    return report


def check_tail_coverage(
    features: pd.DataFrame, df: pd.DataFrame, days: int = 7, max_excess: float = 0.10
) -> pd.DataFrame:
    """(6) No feature may go dark at the end of the stream.

    Blocked/snapshot features assign values to `[boundary_i, boundary_{i+1})`.
    If the last boundary lands before the final row, the trailing rows are
    never assigned and silently keep NaN. Nothing else here catches that: the
    stream ends inside the *test* period, so the affected rows carry no labels,
    sit in no validation fold, and never reach the truncation test -- which
    rebuilds on a stream truncated mid-train and so has its own, different tail.

    Compares each feature's NaN rate over the last `days` of the stream against
    its rate everywhere else, and fails on any excess above `max_excess`.
    Legitimate variation measured here is 0.005, and a single orphaned day
    inside a 7-day tail shows as 0.14, so the threshold sits between them.
    """
    ts = df[C.TIME_COL]
    tail = (ts > ts.max() - pd.Timedelta(days=days)).to_numpy()
    if tail.sum() < 100:
        return pd.DataFrame(columns=["feature", "nan_tail", "nan_rest", "excess"])

    rows = []
    for col in features.columns:
        v = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
        na = ~np.isfinite(v)
        a, b = float(na[tail].mean()), float(na[~tail].mean())
        rows.append({"feature": col, "nan_tail": a, "nan_rest": b, "excess": a - b})

    report = pd.DataFrame(rows).sort_values("excess", ascending=False)
    bad = report[report.excess > max_excess]
    if len(bad):
        raise LeakageError(
            f"feature(s) go missing in the last {days} days of the stream -- "
            "a blocked/snapshot join is not covering the trailing period:" + chr(10)
            + bad.to_string(index=False)
        )
    return report


def check_no_label_in_features(features: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """(1b) No single feature may be near-perfect on its own.

    A lone feature with AP above ~0.85 is a leak, not a discovery. The best
    honest feature measured here is `amount_bdt` at AP 0.217.
    """
    from .validation import single_feature_ap

    mask = np.isfinite(y)
    rows = []
    for col in features.columns:
        v = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
        a, _ = single_feature_ap(v[mask], y[mask])
        rows.append({"feature": col, "ap": a})
    report = pd.DataFrame(rows).sort_values("ap", ascending=False)
    hot = report[report.ap > 0.85]
    if len(hot):
        raise LeakageError(
            "feature(s) individually near-perfect -- almost certainly leakage:\n"
            + hot.to_string(index=False)
        )
    return report


def run_all(features: pd.DataFrame, df: pd.DataFrame, build_fn=None) -> dict:
    """Run every check. `build_fn` enables the truncation test."""
    out = {}
    check_no_banned_columns(features)

    hist_cols = [
        c
        for c in features.columns
        if c.startswith(("cust_amt_prev", "amt_ratio", "amt_z"))
        and not c.endswith(("_100", "_1000", "_int"))
    ]
    check_first_row_is_null(features, df, C.CUSTOMER, hist_cols)

    y = df[C.TARGET].to_numpy(dtype=float)
    out["single_feature_ap"] = check_no_label_in_features(features, y)
    out["boundary"] = check_boundary_continuity(features, df)

    if build_fn is not None:
        out["truncation"] = check_truncation_invariance(
            build_fn, df, cutoff=pd.Timestamp("2026-05-01")
        )
    return out
