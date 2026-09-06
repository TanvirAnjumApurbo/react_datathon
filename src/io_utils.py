"""Loading, cleaning, and construction of the combined time-sorted stream.

The combined stream is the spine of the whole pipeline. The competition rules
explicitly permit it:

    "Feature computation that uses only the raw, non-target columns of test.csv
     in a strictly-past-only way is fine (e.g., a device's known transaction
     history can include test-period rows that occurred earlier than the row
     being scored)"

Building features on train alone would starve every test row of its within-test
history and make test features systematically different from train features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _read_raw(path, has_target: bool) -> pd.DataFrame:
    dtypes = dict(C.RAW_DTYPES)
    if has_target:
        dtypes[C.TARGET] = "int8"
    df = pd.read_csv(path, dtype=dtypes, parse_dates=[C.TIME_COL])
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv and test.csv with explicit dtypes."""
    train = _read_raw(C.RAW_TRAIN, has_target=True)
    test = _read_raw(C.RAW_TEST, has_target=False)
    return train, test


def build_stream(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Concatenate train+test into one deterministically time-sorted frame.

    `fraud` is NaN for test rows. Sorting is by (timestamp, transaction_id):
    there are 20,444 duplicate timestamps in train and 8,587 in test, so the id
    tie-break is required for run-to-run reproducibility.
    """
    train = train.copy()
    test = test.copy()
    train["is_test"] = False
    test["is_test"] = True
    test[C.TARGET] = np.nan

    df = pd.concat([train, test], ignore_index=True)
    df[C.TARGET] = df[C.TARGET].astype("float32")

    df = df.sort_values([C.TIME_COL, C.ID_COL], kind="mergesort").reset_index(drop=True)
    df["row_order"] = np.arange(len(df), dtype=np.int64)
    # Convert to second resolution *explicitly*. Do not divide a raw int64 view
    # by 10**9: pandas 3 stores this column as datetime64[us], so that idiom
    # silently yields kiloseconds and every "1h" window becomes 41.7 days.
    df["ts_epoch"] = df[C.TIME_COL].astype("datetime64[s]").astype("int64")
    return df


def clean_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit-category missing handling + integer entity codes.

    Missingness in merchant_category / device_type / location is MCAR here
    (fraud rate on NA rows 0.0199 / 0.0193 / 0.0170 vs a 0.0176 base rate), so
    mode-imputation would destroy the "field was absent" fact and buy nothing.
    We keep it as its own level and emit an indicator.
    """
    df = df.copy()

    for col in C.NULLABLE_CATS:
        df[f"{col}_was_missing"] = df[col].isna().astype(np.int8)

    for col in C.LOW_CARD_CATS:
        df[col] = df[col].fillna(C.NA_TOKEN).astype("category")

    # Integer codes over the *combined* stream so train and test share a
    # vocabulary. These are grouping keys only -- never model features.
    for col in C.HIGH_CARD_CATS:
        df[f"{col}_code"] = pd.factorize(df[col], sort=True)[0].astype(np.int32)

    return df


def add_implied_signup(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row implied signup day, in whole days since the epoch.

    This is a *raw* per-row quantity, not a customer-level summary: it uses
    only this row's own timestamp and account_age_days. Reconciling it against
    the customer's history is a feature, and lives in features/temporal.py so
    the truncation leakage test covers it.

    (An earlier version summarised this per customer with a groupby median over
    the whole group. That silently read the customer's future rows, and the
    truncation test could not see it because it ran before the function under
    test. Keeping the derivation past-only *and* inside the tested path is the
    fix for both problems.)
    """
    df = df.copy()
    implied = df[C.TIME_COL].dt.normalize() - pd.to_timedelta(df["account_age_days"], unit="D")
    df["implied_signup_day"] = (
        implied.astype("datetime64[s]").astype("int64") // 86_400
    ).astype(np.int32)
    return df


def validate_stream(df: pd.DataFrame) -> None:
    """Sanity assertions on the cleaned stream. Fail loudly, fail early."""
    assert df[C.ID_COL].is_unique, "duplicate transaction_id"
    assert df[C.TIME_COL].is_monotonic_increasing, "stream is not time-sorted"
    assert (df["amount_bdt"] > 0).all(), "non-positive amount_bdt"
    assert (df["account_age_days"] >= 0).all(), "negative account_age_days"

    tr_max = df.loc[~df["is_test"], C.TIME_COL].max()
    te_min = df.loc[df["is_test"], C.TIME_COL].min()
    assert tr_max < te_min, f"train/test overlap: {tr_max} >= {te_min}"

    assert df.loc[df["is_test"], C.TARGET].isna().all(), "test rows carry a label"
    assert df.loc[~df["is_test"], C.TARGET].notna().all(), "train rows missing a label"

    n_train = int((~df["is_test"]).sum())
    n_test = int(df["is_test"].sum())
    assert n_train == 731_942, f"unexpected train row count {n_train}"
    assert n_test == 262_648, f"unexpected test row count {n_test}"

    # ts_epoch must be in *seconds*. A datetime64[us] column divided by 10**9
    # yields kiloseconds and silently rescales every time window by 1000x, so
    # pin the units against the known 258-day span.
    span_days = (df["ts_epoch"].iloc[-1] - df["ts_epoch"].iloc[0]) / 86_400
    assert 250 < span_days < 262, f"ts_epoch is not in seconds (span={span_days:.2f}d)"

    # implied signup day must land in the dataset era, not 1970.
    sd = pd.to_datetime(df["implied_signup_day"] * 86_400, unit="s")
    assert sd.min() > pd.Timestamp("2000-01-01"), "implied_signup_day unit error"
    assert sd.max() <= df[C.TIME_COL].max(), "implied signup in the future"


def get_stream() -> pd.DataFrame:
    """Full load -> clean -> validate path used by every stage."""
    train, test = load_raw()
    df = build_stream(train, test)
    df = clean_stream(df)
    df = add_implied_signup(df)
    validate_stream(df)
    return df
