"""Entity novelty, sharing, and diversity.

Measured on train (base rate 1.76%):

    new device_type for a known customer   ->  9.5% fraud (5.4x)
    new device_id   for a known customer   ->  8.8% fraud (5.0x)
    new location    for a known customer   ->  6.3% fraud (3.6x)
    device used by >=16 prior customers    -> 26.8% fraud (15.2x)

One idea deliberately *not* built as designed: "teleport speed". A location
change gives ~2x lift whether it happened within the hour (1.67x) or after more
than a day (1.99x), so the elapsed time carries no extra information here. The
change flag and the time-since-change are kept; a speed ratio is not.

Windowed distinct counts are expressed as *new pairs formed in the window*
rather than distinct-entities-in-window. That is exact, cheap, and a sharper
signal: a device that picked up 9 new customers in 24h is a farm spinning up,
whereas a distinct count would fire equally for 9 regulars returning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ._windows import WindowIndex, first_occurrence_flag, new_pairs_in_window, running_nunique


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where((b == 0) | ~np.isfinite(b), np.nan, b)
    return a / b


# customer-side novelty: which raw column, and the short feature name
_NOVELTY = {
    "dev": f"{C.DEVICE}_code",
    "merch": f"{C.MERCHANT}_code",
    "loc": "location",
    "dtype": "device_type",
    "mcat": "merchant_category",
    "pay": "payment_method",
    "ttype": "transaction_type",
}


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    epoch = df["ts_epoch"].to_numpy()
    cust = df[f"{C.CUSTOMER}_code"].to_numpy().astype(np.int64)
    dev = df[f"{C.DEVICE}_code"].to_numpy().astype(np.int64)
    merch = df[f"{C.MERCHANT}_code"].to_numpy().astype(np.int64)

    cust_prior_n = WindowIndex(cust, epoch).prior_count()
    has_history = cust_prior_n > 0

    # --- novelty of (customer, entity) pairs ------------------------------
    for short, col in _NOVELTY.items():
        vals = df[col]
        ent = (
            vals.to_numpy().astype(np.int64)
            if col.endswith("_code")
            else pd.factorize(vals, sort=True)[0].astype(np.int64)
        )
        pair = pd.factorize(cust * 1_000_003 + ent, sort=True)[0].astype(np.int64)
        widx = WindowIndex(pair, epoch)

        prior = widx.prior_count()
        # "New" only means something once the customer has any history at all.
        out[f"cust_new_{short}"] = ((prior == 0) & has_history).astype(np.int8)
        out[f"cust_{short}_prior_n"] = prior
        out[f"cust_{short}_dt"] = widx.prev_gap(1)
        # Share of the customer's life spent with this entity.
        out[f"cust_{short}_share"] = _safe_div(
            prior.astype(np.float64), cust_prior_n.astype(np.float64)
        )

    # --- device sharing: the account-farm signal --------------------------
    out["dev_prior_ncust"] = running_nunique(dev, cust)
    out["dev_prior_nloc"] = running_nunique(
        dev, pd.factorize(df["location"], sort=True)[0].astype(np.int64)
    )
    for w in ("24h", "168h"):
        out[f"dev_new_cust_{w}"] = new_pairs_in_window(dev, cust, epoch, C.WINDOWS_S[w])
    dev_prior_n = WindowIndex(dev, epoch).prior_count()
    # A device whose traffic is mostly first-time customers is not a phone.
    out["dev_cust_churn"] = _safe_div(
        out["dev_prior_ncust"].to_numpy().astype(np.float64),
        dev_prior_n.astype(np.float64),
    )

    # --- merchant sharing -------------------------------------------------
    out["merch_prior_ncust"] = running_nunique(merch, cust)
    out["merch_prior_ndev"] = running_nunique(merch, dev)

    # --- customer diversity ----------------------------------------------
    out["cust_prior_ndev"] = running_nunique(cust, dev)
    out["cust_prior_nmerch"] = running_nunique(cust, merch)
    out["cust_prior_nloc"] = running_nunique(
        cust, pd.factorize(df["location"], sort=True)[0].astype(np.int64)
    )
    out["cust_prior_nmcat"] = running_nunique(
        cust, pd.factorize(df["merchant_category"], sort=True)[0].astype(np.int64)
    )
    for w in ("24h", "168h"):
        out[f"cust_new_dev_{w}"] = new_pairs_in_window(cust, dev, epoch, C.WINDOWS_S[w])
        out[f"cust_new_loc_{w}"] = new_pairs_in_window(
            cust,
            pd.factorize(df["location"], sort=True)[0].astype(np.int64),
            epoch,
            C.WINDOWS_S[w],
        )
    # Devices per transaction: a customer cycling handsets.
    out["cust_dev_per_txn"] = _safe_div(
        out["cust_prior_ndev"].to_numpy().astype(np.float64),
        cust_prior_n.astype(np.float64),
    )

    # --- location movement ------------------------------------------------
    loc_code = pd.factorize(df["location"], sort=True)[0].astype(np.int64)
    widx_cust = WindowIndex(cust, epoch)
    prev_loc = widx_cust.prev_value(loc_code.astype(np.float64), 1)
    changed = (
        np.isfinite(prev_loc) & (prev_loc != loc_code.astype(np.float64))
    ).astype(np.int8)
    out["loc_changed_vs_prev"] = changed
    # Time since the customer last changed location (not a "speed" -- see docstring).
    chg_series = pd.Series(changed.astype(bool))
    last_change_epoch = (
        pd.Series(np.where(changed == 1, epoch, np.nan))
        .groupby(pd.Series(cust), sort=False)
        .ffill()
        .to_numpy()
    )
    out["time_since_loc_change"] = epoch - last_change_epoch

    # --- entity age (past-only first-seen) --------------------------------
    # Raw age is monotone in calendar time: every device is older in the test
    # period than in the train period, so a tree that learns "age < 200 => X"
    # extrapolates off a cliff at the boundary -- the same failure mode that
    # gets transaction_id banned. Each raw age is therefore paired with a
    # fraction-of-observable-history version, which is drift-stable.
    stream_age_days = np.maximum((epoch - epoch[0]) / 86_400.0, 1e-6)
    for short, code in (("dev", dev), ("merch", merch)):
        first = first_occurrence_flag(code).astype(bool)
        first_epoch = (
            pd.Series(np.where(first, epoch, np.nan))
            .groupby(pd.Series(code), sort=False)
            .ffill()
            .to_numpy()
        )
        age_days = (epoch - first_epoch) / 86_400.0
        out[f"{short}_age_days"] = age_days
        out[f"{short}_age_frac"] = np.clip(age_days / stream_age_days, 0.0, 1.0)
        out[f"{short}_is_first_seen"] = first.astype(np.int8)

    # Same treatment for the customer's own tenure inside the stream.
    cust_first = first_occurrence_flag(cust).astype(bool)
    cust_first_epoch = (
        pd.Series(np.where(cust_first, epoch, np.nan))
        .groupby(pd.Series(cust), sort=False)
        .ffill()
        .to_numpy()
    )
    cust_age_days = (epoch - cust_first_epoch) / 86_400.0
    out["cust_stream_age_days"] = cust_age_days
    out["cust_stream_age_frac"] = np.clip(cust_age_days / stream_age_days, 0.0, 1.0)
    # Activity rate normalised by observed tenure -- comparable across periods.
    out["cust_txn_per_active_day"] = _safe_div(
        cust_prior_n.astype(np.float64), np.maximum(cust_age_days, 1.0)
    )

    int_cols = [c for c in out.columns if out[c].dtype in (np.int64, np.int32)]
    out = out.astype({c: np.int32 for c in int_cols})
    float_cols = [c for c in out.columns if out[c].dtype == np.float64]
    return out.astype({c: C.FLOAT_DTYPE for c in float_cols})
