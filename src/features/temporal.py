"""Time-of-day / calendar features, including per-customer circular rhythm.

Measured signal (train): hours 00:00-04:59 carry 5.5-9.8x fraud lift, peaking
at 17.25% fraud at 03:00 against a 1.76% base rate. That global nocturnal
pattern is the dominant temporal effect; the per-customer von Mises features
(Bahnsen et al. 2016) are much weaker here (AP 0.026 vs 0.046 for the plain
night flag) because fraud in this dataset is globally nocturnal rather than
anomalous relative to each customer's own rhythm. They are built anyway --
cheap, and they let the model separate "3am for everyone" from "3am for a
customer who never transacts at night" -- but they are not tuned.

No absolute time index is emitted. transaction_id and epoch seconds are
monotone across the train/test boundary and would let a tree extrapolate off a
cliff; see config.BANNED_FEATURES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ._windows import WindowIndex


def build(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[C.TIME_COL]
    out = pd.DataFrame(index=df.index)

    hour = ts.dt.hour.to_numpy()
    minute = ts.dt.minute.to_numpy()
    hour_frac = hour + minute / 60.0

    out["hour"] = hour.astype(np.int8)
    out["minute_of_day"] = (hour * 60 + minute).astype(np.int16)
    out["dayofweek"] = ts.dt.dayofweek.to_numpy().astype(np.int8)
    out["day_of_month"] = ts.dt.day.to_numpy().astype(np.int8)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(np.int8)

    # Cyclical encodings so 23:00 and 00:00 sit next to each other.
    theta = 2 * np.pi * hour_frac / 24.0
    out["hour_sin"] = np.sin(theta)
    out["hour_cos"] = np.cos(theta)
    dow_theta = 2 * np.pi * out["dayofweek"].to_numpy() / 7.0
    out["dow_sin"] = np.sin(dow_theta)
    out["dow_cos"] = np.cos(dow_theta)

    # The single highest-value temporal flag.
    out["is_night"] = ((hour >= 0) & (hour < 5)).astype(np.int8)
    out["is_deep_night"] = ((hour >= 1) & (hour < 5)).astype(np.int8)

    # --- per-customer circular rhythm (von Mises style) -------------------
    cust = df[f"{C.CUSTOMER}_code"].to_numpy()
    epoch = df["ts_epoch"].to_numpy()
    widx = WindowIndex(cust, epoch)

    cos_mean = widx.expanding_prev(np.cos(theta), "mean")
    sin_mean = widx.expanding_prev(np.sin(theta), "mean")
    mu = np.arctan2(sin_mean, cos_mean)
    # Resultant length: 0 = customer transacts at all hours, 1 = fixed hour.
    R = np.sqrt(cos_mean**2 + sin_mean**2)
    ang_dev = np.abs(np.arctan2(np.sin(theta - mu), np.cos(theta - mu)))

    out["cust_time_concentration"] = R
    out["cust_ang_dev"] = ang_dev
    # Deviation only means something for a customer with a stable rhythm.
    out["cust_ang_dev_weighted"] = ang_dev * R

    # --- how unusual is this hour globally, by volume (not by label) ------
    # Computed as an expanding past-only share. A full-stream histogram would
    # be label-free but still forward-looking, and the rules bind on time as
    # well as on the target: "may only use information strictly before t".
    h_ser = pd.Series(hour)
    prior_same_hour = h_ser.groupby(h_ser, sort=False).cumcount().to_numpy()
    prior_total = np.arange(len(df), dtype=np.int64)
    out["hour_volume_share"] = np.where(
        prior_total > 0, prior_same_hour / np.maximum(prior_total, 1), np.nan
    )

    # --- account tenure ---------------------------------------------------
    out["account_age_days"] = df["account_age_days"].to_numpy().astype(np.float64)
    out["account_age_log"] = np.log1p(out["account_age_days"].to_numpy())

    # Reconcile this row's implied signup day against the one the customer's
    # *earlier* rows established. Past-only: the first row of any customer has
    # nothing to compare against and gets NaN.
    implied = df["implied_signup_day"].to_numpy().astype(np.float64)
    baseline = widx.expanding_prev(implied, "median")
    out["signup_inconsistency_d"] = np.abs(implied - baseline)
    out["days_since_signup"] = out["account_age_days"].to_numpy()

    if not C.USE_SIGNUP_INCONSISTENCY:
        # See config.USE_SIGNUP_INCONSISTENCY for why this is switchable.
        out = out.drop(columns=["signup_inconsistency_d"])

    float_cols = [c for c in out.columns if out[c].dtype == np.float64]
    return out.astype({c: C.FLOAT_DTYPE for c in float_cols})
