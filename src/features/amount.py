"""Amount features, absolute and relative to the customer's own history.

`amount_bdt` is the strongest single feature measured (AP 0.217 on a held-out
future month, 12.7x the 0.0170 base rate) -- and also the most drift-exposed.
Train P99.9 is 42,007 BDT but test P99.9 is only 24,342, and P(amount > 10,000)
falls from 0.69% to 0.45%, while the night-hour share barely moves. The fraud
amount archetype genuinely shifts into the test period.

So raw amount is always paired with relative forms:
  * against the customer's own past (ratio / z / vs-max), which re-centres per
    customer;
  * against a trailing global window (percentile rank), which re-centres as the
    population distribution drifts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ._windows import WindowIndex

_N_QUANTILES = 128


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where((b == 0) | ~np.isfinite(b), np.nan, b)
    return a / b


def _trailing_rank(
    values: np.ndarray, epoch: np.ndarray, day_idx: np.ndarray, window_d: int
) -> np.ndarray:
    """Percentile rank of each amount within a trailing `window_d`-day window.

    Rather than recomputing a rank per row, we refresh a quantile grid once per
    calendar day from data strictly *before that day begins*, then rank every
    row of the day against it. Past-only by construction, and 258 grid builds
    instead of ~1M window scans.

    Within-day staleness is immaterial for a 7- or 30-day window, and it makes
    the feature behave identically on train and test days.
    """
    n = len(values)
    window_s = window_d * 86_400
    qs = np.linspace(0.0, 1.0, _N_QUANTILES)

    n_days = int(day_idx.max()) + 1
    # Stream is time-sorted, so each day occupies one contiguous slice.
    day_start_pos = np.searchsorted(day_idx, np.arange(n_days + 1), side="left")
    day_start_epoch = np.empty(n_days, dtype=np.int64)
    for d in range(n_days):
        lo = day_start_pos[d]
        day_start_epoch[d] = epoch[lo] if lo < n else epoch[-1]

    # First row inside the trailing window for each day boundary.
    win_lo = np.searchsorted(epoch, day_start_epoch - window_s, side="left")

    out = np.full(n, np.nan)
    for d in range(n_days):
        lo, hi = win_lo[d], day_start_pos[d]
        if hi - lo < 100:  # not enough history to rank against
            continue
        grid = np.quantile(values[lo:hi], qs)
        rows = slice(day_start_pos[d], day_start_pos[d + 1])
        out[rows] = np.searchsorted(grid, values[rows], side="right") / _N_QUANTILES
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    amt = df["amount_bdt"].to_numpy(dtype=np.float64)
    epoch = df["ts_epoch"].to_numpy()
    cust = df[f"{C.CUSTOMER}_code"].to_numpy()

    out["amount_bdt"] = amt
    out["log_amount"] = np.log1p(amt)

    # --- vs the customer's own past ---------------------------------------
    widx = WindowIndex(cust, epoch)
    prev_mean = widx.expanding_prev(amt, "mean")
    prev_std = widx.expanding_prev(amt, "std")
    prev_max = widx.expanding_prev(amt, "max")
    prev_min = widx.expanding_prev(amt, "min")
    prev_median = widx.expanding_prev(amt, "median")

    out["cust_amt_prev_mean"] = prev_mean
    out["cust_amt_prev_std"] = prev_std
    out["cust_amt_prev_max"] = prev_max
    out["amt_ratio_mean"] = _safe_div(amt, prev_mean)
    out["amt_ratio_median"] = _safe_div(amt, prev_median)
    out["amt_ratio_max"] = _safe_div(amt, prev_max)
    out["amt_ratio_min"] = _safe_div(amt, prev_min)
    out["amt_z"] = _safe_div(amt - prev_mean, prev_std)
    out["log_amt_ratio_mean"] = np.log1p(amt) - np.log1p(prev_mean)

    out["amt_ratio_last1"] = _safe_div(amt, widx.prev_value(amt, 1))
    out["amt_ratio_last5"] = _safe_div(amt, widx.rolling_mean_last_k(amt, 5))
    out["amt_ratio_last20"] = _safe_div(amt, widx.rolling_mean_last_k(amt, 20))

    # --- vs the customer's own past *within this context* -----------------
    # 30,000 BDT of electronics is not 30,000 BDT of mobile topup.
    for ctx in ["merchant_category", "transaction_type", "payment_method"]:
        ctx_code = pd.factorize(df[ctx], sort=True)[0].astype(np.int64)
        pair = pd.factorize(cust.astype(np.int64) * 100 + ctx_code, sort=True)[0]
        widx_ctx = WindowIndex(pair.astype(np.int64), epoch)
        m = widx_ctx.expanding_prev(amt, "mean")
        s = widx_ctx.expanding_prev(amt, "std")
        short = ctx.replace("merchant_", "").replace("transaction_", "")[:5]
        out[f"amt_ratio_mean_by_{short}"] = _safe_div(amt, m)
        out[f"amt_z_by_{short}"] = _safe_div(amt - m, s)
        out[f"cust_{short}_prior_n"] = widx_ctx.prior_count()

    # --- drift-robust: percentile rank in a trailing global window --------
    day_idx = (
        df[C.TIME_COL].dt.normalize().rank(method="dense").to_numpy().astype(np.int64) - 1
    )
    for w in C.AMOUNT_RANK_WINDOWS_D:
        out[f"amt_rank_{w}d"] = _trailing_rank(amt, epoch, day_idx, w)

    # --- round-number flags (weak but free) -------------------------------
    a2 = np.round(amt, 2)
    out["amt_is_int"] = (a2 == np.round(a2)).astype(np.int8)
    out["amt_round_100"] = (np.mod(a2, 100) == 0).astype(np.int8)
    out["amt_round_1000"] = (np.mod(a2, 1000) == 0).astype(np.int8)

    float_cols = [c for c in out.columns if out[c].dtype == np.float64]
    return out.astype({c: C.FLOAT_DTYPE for c in float_cols})
