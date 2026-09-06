"""Velocity: how fast is this entity transacting, relative to its own norm.

The strongest block in the pipeline. Measured on train:

    customer's previous txn < 1 min ago  ->  86.0% fraud (48.9x lift)
    device's previous txn  < 10 s ago    ->  83.0% fraud (47.1x lift)

Merchant velocity, by contrast, is worthless here (a <10s merchant gap gives
0.69x lift) because the merchants are high-traffic aggregators -- the top 10
carry 57.8% of all rows. It is built anyway so the model can confirm that for
itself, but it is expected to rank near the bottom.

Windows follow the Whitrow / Bahnsen transaction-aggregation convention
(1h / 6h / 24h / 72h / 168h). All counts are strictly past: they exclude the
row being scored. See _windows.WindowIndex for the guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ._windows import WindowIndex


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where((b == 0) | ~np.isfinite(b), np.nan, b)
    return a / b


# key name -> (column holding the group code, which windows to aggregate)
_KEYS = {
    "cust": (f"{C.CUSTOMER}_code", ["1h", "6h", "24h", "72h", "168h"]),
    "dev": (f"{C.DEVICE}_code", ["1h", "6h", "24h", "168h"]),
    "merch": (f"{C.MERCHANT}_code", ["1h", "24h"]),
    "loc": ("location_code", ["1h", "24h"]),
}


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    epoch = df["ts_epoch"].to_numpy()
    amt = df["amount_bdt"].to_numpy(dtype=np.float64)

    codes = {
        f"{C.CUSTOMER}_code": df[f"{C.CUSTOMER}_code"].to_numpy(),
        f"{C.DEVICE}_code": df[f"{C.DEVICE}_code"].to_numpy(),
        f"{C.MERCHANT}_code": df[f"{C.MERCHANT}_code"].to_numpy(),
        "location_code": pd.factorize(df["location"], sort=True)[0].astype(np.int64),
    }

    for name, (col, windows) in _KEYS.items():
        widx = WindowIndex(codes[col], epoch)

        out[f"{name}_prior_n"] = widx.prior_count()
        gap1 = widx.prev_gap(1)
        out[f"{name}_dt"] = gap1
        out[f"{name}_log_dt"] = np.log1p(gap1)

        if name in ("cust", "dev"):
            out[f"{name}_dt2"] = widx.prev_gap(2)
            out[f"{name}_dt3"] = widx.prev_gap(3)
            # Burst relative to this entity's own normal cadence.
            typical = widx.expanding_prev(np.nan_to_num(gap1, nan=0.0), "median")
            out[f"{name}_dt_vs_typical"] = _safe_div(gap1, typical)

        for w in windows:
            wsec = C.WINDOWS_S[w]
            cnt = widx.count_in_window(wsec)
            out[f"{name}_n_{w}"] = cnt
            if name in ("cust", "dev"):
                s = widx.sum_in_window(amt, wsec)
                out[f"{name}_amt_sum_{w}"] = s
                out[f"{name}_amt_mean_{w}"] = _safe_div(s, cnt.astype(np.float64))

        # Burst-vs-baseline: short window against long window.
        if name == "cust":
            out["cust_burst_1h_24h"] = _safe_div(
                out["cust_n_1h"].to_numpy().astype(np.float64) ,
                out["cust_n_24h"].to_numpy().astype(np.float64),
            )
            out["cust_burst_24h_168h"] = _safe_div(
                out["cust_n_24h"].to_numpy().astype(np.float64),
                out["cust_n_168h"].to_numpy().astype(np.float64),
            )
            # Rate normalised by how long the customer has been active.
            out["cust_txn_per_day"] = _safe_div(
                out["cust_prior_n"].to_numpy().astype(np.float64),
                np.maximum(df["days_since_signup"].to_numpy(), 1.0)
                if "days_since_signup" in df
                else np.maximum(df["account_age_days"].to_numpy(), 1.0),
            )
        if name == "dev":
            out["dev_burst_1h_24h"] = _safe_div(
                out["dev_n_1h"].to_numpy().astype(np.float64),
                out["dev_n_24h"].to_numpy().astype(np.float64),
            )

    # --- the (customer, device) pair: "this person on this phone" ---------
    pair = pd.factorize(
        codes[f"{C.CUSTOMER}_code"].astype(np.int64) * 1_000_003
        + codes[f"{C.DEVICE}_code"].astype(np.int64),
        sort=True,
    )[0].astype(np.int64)
    widx_pair = WindowIndex(pair, epoch)
    out["custdev_prior_n"] = widx_pair.prior_count()
    out["custdev_dt"] = widx_pair.prev_gap(1)
    out["custdev_n_24h"] = widx_pair.count_in_window(C.WINDOWS_S["24h"])
    # What share of the customer's recent activity ran through this device?
    out["custdev_share_24h"] = _safe_div(
        out["custdev_n_24h"].to_numpy().astype(np.float64),
        out["cust_n_24h"].to_numpy().astype(np.float64),
    )

    int_cols = [c for c in out.columns if out[c].dtype == np.int64]
    out = out.astype({c: np.int32 for c in int_cols})
    float_cols = [c for c in out.columns if out[c].dtype == np.float64]
    return out.astype({c: C.FLOAT_DTYPE for c in float_cols})
