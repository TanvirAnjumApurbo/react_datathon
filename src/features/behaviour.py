"""Habit, repetition, movement, and account youth.

Four ideas that the amount/velocity/entity blocks do not express, each chosen
because it says something those blocks cannot:

**Habit (entropy and surprise).** A customer who always shops in two categories
has a low-entropy history; a customer who shops everywhere has a high-entropy
one. The same new category means very different things in those two cases.
`*_entropy` measures the habit and `*_surprise` measures how far this
particular transaction sits outside it. The surprise term is the sharper of the
two: it is the self-information of the observed category under the customer's
own past distribution, so a first-ever category scores high for a creature of
habit and unremarkable for someone who is all over the map.

**Repetition (duplicates).** Card-testing and retry fraud produce near-identical
transactions in quick succession. Velocity counts how *many* transactions
happened recently; this counts how many of them were *the same* -- same
merchant, same amount. Those are different signals, and a burst of identical
attempts is the more specific one.

**Movement (travel).** `entity.py` already flags a location change and the time
since the last one. What it cannot say is how *fast* the changes are coming. A
customer who switched location four times in a day is a different proposition
from one who moved once, and both have `loc_changed_vs_prev = 1`.

**Account youth.** New-account fraud is a documented typology, and the
interaction is the point: a large amount is ordinary on a two-year-old account
and not on a two-week-old one. `account_age_days` is also the single strongest
train/test separator in the raw data (adversarial AUC 0.553, PSI 0.221, mean
459 -> 513 days) because it counts upward for every returning customer, so it
gets a trailing-percentile companion that re-centres as the population ages.

Everything routes through `_windows.WindowIndex` or a past-only cumulative sum,
so `check_truncation_invariance` covers all of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ._windows import WindowIndex, first_occurrence_flag
from .amount import _trailing_rank

# Categorical histories worth summarising per customer. `location` and
# `merchant_category` are the behavioural ones; the hour bucket captures a
# customer's routine in time.
ENTROPY_COLS = ["merchant_category", "location", "device_type"]

# Windows for the repetition and movement counters.
DUP_WINDOWS = ["1h", "24h", "168h"]
TRAVEL_WINDOWS = ["24h", "168h"]

YOUNG_ACCOUNT_D = 30.0


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where((b == 0) | ~np.isfinite(b), np.nan, b)
    return a / b


def _expanding_entropy(
    cust: np.ndarray, cat: np.ndarray, n_cat: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shannon entropy of a customer's prior category distribution, and the
    self-information of the current category under it.

    Both from one identity. With `c_k` the customer's prior count of category
    `k` and `N = sum_k c_k`,

        H = -sum_k (c_k/N) log(c_k/N) = log(N) - (1/N) sum_k c_k log(c_k)

    so the only running quantity needed is `S = sum_k c_k log c_k`. When a row
    of category `k` arrives, `S` changes by exactly

        (c_k + 1) log(c_k + 1) - c_k log(c_k)

    which depends only on that row's own prior pair count. So the per-row
    increment is computable directly from `WindowIndex.prior_count()` on the
    (customer, category) pair, and `S` before a row is a grouped cumulative sum
    of increments minus the row's own -- strictly past, one vectorised pass, no
    per-customer dictionary.

    Returns (normalised entropy, surprise). The entropy is divided by
    `log(min(N, n_cat))`, its maximum achievable value given how much history
    exists, so a customer with three transactions is comparable to one with
    three hundred instead of being penalised for having a short history.
    """
    n = len(cust)
    pair = (cust.astype(np.int64) * np.int64(n_cat + 1)) + cat.astype(np.int64)

    # Prior counts: c_k for this row's category, and N over all categories.
    c_k = WindowIndex(pair, np.arange(n, dtype=np.int64)).prior_count().astype(np.float64)
    N = WindowIndex(cust.astype(np.int64), np.arange(n, dtype=np.int64)).prior_count()
    N = N.astype(np.float64)

    def xlogx(x):
        return np.where(x > 0, x * np.log(np.maximum(x, 1e-12)), 0.0)

    delta = xlogx(c_k + 1.0) - xlogx(c_k)
    s = pd.Series(delta)
    S = s.groupby(pd.Series(cust), sort=False).cumsum().to_numpy() - delta

    with np.errstate(invalid="ignore", divide="ignore"):
        H = np.log(np.maximum(N, 1e-12)) - _safe_div(S, N)
        H = np.where(N > 0, H, np.nan)
        # Normalise by the entropy a history of this length could reach.
        H_max = np.log(np.maximum(np.minimum(N, float(n_cat)), 1.0 + 1e-12))
        H_norm = np.where(N > 1, H / H_max, np.nan)

        # Laplace-smoothed self-information of the observed category. Defined
        # even on a first-ever row, where it is simply log(n_cat).
        p = (c_k + 1.0) / (N + float(n_cat))
        surprise = -np.log(p)

    return H_norm.astype(np.float64), surprise.astype(np.float64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    n = len(df)
    epoch = df["ts_epoch"].to_numpy()
    amt = df["amount_bdt"].to_numpy(dtype=np.float64)
    cust = df[f"{C.CUSTOMER}_code"].to_numpy().astype(np.int64)
    merch = df[f"{C.MERCHANT}_code"].to_numpy().astype(np.int64)
    dev = df[f"{C.DEVICE}_code"].to_numpy().astype(np.int64)
    age = df["account_age_days"].to_numpy(dtype=np.float64)

    widx_cust = WindowIndex(cust, epoch)

    # ---- habit: entropy and surprise -------------------------------------
    for col in ENTROPY_COLS:
        codes = pd.factorize(df[col], sort=True)[0].astype(np.int64)
        k = int(codes.max()) + 1
        H, surprise = _expanding_entropy(cust, codes, k)
        short = col.replace("merchant_", "").replace("device_", "d")[:5]
        out[f"cust_{short}_entropy"] = H
        out[f"cust_{short}_surprise"] = surprise

    # The customer's routine in time, on the same footing.
    hour_bucket = np.clip(df[C.TIME_COL].dt.hour.to_numpy() // 3, 0, 7).astype(np.int64)
    H, surprise = _expanding_entropy(cust, hour_bucket, 8)
    out["cust_hour_entropy"] = H
    out["cust_hour_surprise"] = surprise

    # ---- repetition: duplicates and near-duplicates ----------------------
    # Exact repeat = same customer, same merchant, same amount to the paisa.
    amt_key = np.round(amt * 100).astype(np.int64)
    exact = pd.factorize(
        pd.MultiIndex.from_arrays([cust, merch, amt_key]), sort=False
    )[0].astype(np.int64)
    widx_exact = WindowIndex(exact, epoch)
    out["dup_exact_prior_n"] = widx_exact.prior_count()
    for w in DUP_WINDOWS:
        col = widx_exact.count_in_window(C.WINDOWS_S[w])
        # Amounts here are continuous to the paisa, so an exact repeat inside
        # an hour never happens: the 1h column is identically zero over all
        # 994,590 rows. A constant column cannot inform a split, so it is
        # dropped rather than shipped as noise.
        if np.ptp(col) == 0:
            continue
        out[f"dup_exact_{w}"] = col

    # Near-repeat = same customer and merchant, amount within the same 1%
    # log bucket. Catches a retry that nudged the amount rather than repeating
    # it exactly, which is the more common evasion.
    amt_bucket = np.round(np.log1p(amt) * 100).astype(np.int64)
    near = pd.factorize(
        pd.MultiIndex.from_arrays([cust, merch, amt_bucket]), sort=False
    )[0].astype(np.int64)
    widx_near = WindowIndex(near, epoch)
    out["dup_near_24h"] = widx_near.count_in_window(C.WINDOWS_S["24h"])
    # Same merchant at all, regardless of amount -- the denominator that makes
    # the two counts above readable as "how repetitive", not just "how busy".
    widx_cm = WindowIndex(
        pd.factorize(pd.MultiIndex.from_arrays([cust, merch]), sort=False)[0].astype(np.int64),
        epoch,
    )
    cm_24h = widx_cm.count_in_window(C.WINDOWS_S["24h"]).astype(np.float64)
    out["dup_exact_share_24h"] = _safe_div(
        out["dup_exact_24h"].to_numpy().astype(np.float64), cm_24h
    )

    # Same-instant collisions. `count_in_window(0)` resolves to rows of the key
    # sharing this exact timestamp and ordered earlier by transaction_id --
    # which is what a scoring system would have seen.
    out["cust_same_ts_prior"] = widx_cust.count_in_window(0)
    out["dev_same_ts_prior"] = WindowIndex(dev, epoch).count_in_window(0)

    # ---- movement: how fast the location is changing ---------------------
    loc = pd.factorize(df["location"], sort=True)[0].astype(np.int64)
    prev_loc = widx_cust.prev_value(loc.astype(np.float64), 1)
    switched = np.where(
        np.isfinite(prev_loc), (prev_loc != loc.astype(np.float64)).astype(np.float64), np.nan
    )
    sw = np.nan_to_num(switched, nan=0.0)
    for w in TRAVEL_WINDOWS:
        cnt = widx_cust.count_in_window(C.WINDOWS_S[w]).astype(np.float64)
        n_sw = widx_cust.sum_in_window(sw, C.WINDOWS_S[w])
        out[f"loc_switches_{w}"] = n_sw
        # Rate rather than count: a customer with twenty transactions a day is
        # expected to move around more than one with two, and the count alone
        # would call the busy customer suspicious. The rate also stays bounded
        # as the stream lengthens, unlike a lifetime switch count.
        out[f"loc_switch_rate_{w}"] = _safe_div(n_sw, cnt)

    # Speed of the change: a switch is only interesting if it happened fast.
    gap = widx_cust.prev_gap(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["loc_switch_recency"] = np.where(sw > 0, np.log1p(gap), np.nan)
    out["loc_switch_fast"] = ((sw > 0) & (gap < C.WINDOWS_S["1h"])).astype(np.int8)

    # ---- account youth, and its interactions -----------------------------
    out["is_young_account"] = (age <= YOUNG_ACCOUNT_D).astype(np.int8)
    out["log_account_age"] = np.log1p(age)
    # Amount relative to account age, in log space: a big first-week transfer
    # is the signal, not a big transfer.
    out["amt_per_acct_age"] = np.log1p(amt) - np.log1p(age)

    # Trailing percentile of account age against the recent population. Raw
    # account age rises for every returning customer, so its absolute value
    # drifts with the calendar and every test row sits above the train range.
    # The percentile re-centres each day against whoever was transacting
    # recently, which is the same fix `*_age_frac` applies elsewhere.
    day_idx = (
        df[C.TIME_COL].dt.normalize().rank(method="dense").to_numpy().astype(np.int64) - 1
    )
    out["acct_age_rank_30d"] = _trailing_rank(age, epoch, day_idx, 30)

    # The composite the research report calls for: new account + unusual
    # amount + unfamiliar device. Kept as a 0-3 count rather than a single AND
    # so the model can use the partial cases, which are far more common.
    new_dev = first_occurrence_flag(cust, dev).astype(np.int64)
    prev_mean = widx_cust.expanding_prev(amt, "mean")
    big_amt = (_safe_div(amt, prev_mean) > 3.0).astype(np.int64)
    out["new_dev_for_cust"] = new_dev.astype(np.int8)
    out["young_acct_risk"] = (
        out["is_young_account"].to_numpy().astype(np.int64) + big_amt + new_dev
    ).astype(np.int8)

    # Velocity interaction: transactions per day of account life. A one-week
    # account running at ten a day is behaving unlike a one-week account.
    out["txn_rate_vs_acct_age"] = _safe_div(
        widx_cust.count_in_window(C.WINDOWS_S["24h"]).astype(np.float64),
        np.log1p(age) + 1.0,
    )

    float_cols = [c for c in out.columns if out[c].dtype in (np.float64, np.int64)]
    out = out.astype({c: C.FLOAT_DTYPE for c in float_cols})

    # Four independent ideas live in this module, and lumping them into one
    # "behaviour" block would force a single keep-or-drop verdict on all of
    # them. Tagging lets the ablation ask about each separately, which matters
    # because the standalone reads already differ by an order of magnitude:
    # the youth interactions reach ~5x base-rate lift on the tail window while
    # the duplicate counters sit at 1.0x -- exact repeats essentially do not
    # occur once amounts are continuous.
    out.attrs["subblock"] = {c: _subblock_of(c) for c in out.columns}
    return out


def _subblock_of(col: str) -> str:
    if col.startswith("dup_") or col.endswith("_same_ts_prior"):
        return "repetition"
    if col.startswith("loc_switch"):
        return "travel"
    if "entropy" in col or "surprise" in col:
        return "habit"
    return "youth"
