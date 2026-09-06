"""Central configuration for the REACT 2026 fraud pipeline.

Everything that a downstream stage needs to agree on lives here: paths, the
time windows used by the aggregation features, the validation cutoffs, and the
columns that are *forbidden* from ever reaching the model matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_TRAIN = DATA / "train.csv"
RAW_TEST = DATA / "test.csv"
SAMPLE_SUB = DATA / "sample_submission.csv"
PROCESSED = DATA / "processed"
GRAPH_CACHE = PROCESSED / "graph"

for _d in (PROCESSED, GRAPH_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------
# Raw schema
# --------------------------------------------------------------------------
ID_COL = "transaction_id"
TIME_COL = "timestamp"
TARGET = "fraud"

CUSTOMER = "customer_id"
MERCHANT = "merchant_id"
DEVICE = "device_id"

# Categoricals with a small, closed vocabulary (identical in train and test).
LOW_CARD_CATS = [
    "merchant_category",
    "device_type",
    "location",
    "payment_method",
    "transaction_type",
]
# The three that carry missing values (~0.4-0.6%, MCAR).
NULLABLE_CATS = ["merchant_category", "device_type", "location"]
NA_TOKEN = "__NA__"

HIGH_CARD_CATS = [CUSTOMER, MERCHANT, DEVICE]

RAW_DTYPES = {
    ID_COL: "string",
    CUSTOMER: "string",
    MERCHANT: "string",
    DEVICE: "string",
    "amount_bdt": "float64",
    "account_age_days": "int32",
    **{c: "string" for c in LOW_CARD_CATS},
}

# --------------------------------------------------------------------------
# Feature windows
# --------------------------------------------------------------------------
# Whitrow / Bahnsen transaction-aggregation windows, in seconds.
WINDOWS_S = {
    "1h": 3_600,
    "6h": 21_600,
    "24h": 86_400,
    "72h": 259_200,
    "168h": 604_800,
}
# Trailing windows used for the drift-robust amount percentile ranks.
AMOUNT_RANK_WINDOWS_D = [7, 30]

# Bayesian smoothing strength for past-only target encoding.
TE_ALPHA = 100.0
# Half-life (days) for the time-decayed target-encoding variant.
TE_HALFLIFE_D = 30.0

# Graph snapshot cadence. "W" -> 38 snapshots, "D" -> 258.
GRAPH_SNAPSHOT_FREQ = "W"
GRAPH_EMBED_DIM = 32
GNN_EMBED_DIM = 32
GNN_EPOCHS = 30

# --------------------------------------------------------------------------
# Validation (see plan Stage 8)
# --------------------------------------------------------------------------
TRAIN_START = pd.Timestamp("2026-01-01")
TRAIN_END = pd.Timestamp("2026-07-15 23:59:59")
TEST_START = pd.Timestamp("2026-07-16")
TEST_END = pd.Timestamp("2026-09-15 23:59:59")

# The real task is a 62-day forward block immediately after the train cutoff.
# The primary fold reproduces that geometry exactly.
PRIMARY_FOLD = {
    "name": "primary_62d",
    "train_end": pd.Timestamp("2026-05-14 23:59:59"),
    "val_start": pd.Timestamp("2026-05-15"),
    "val_end": pd.Timestamp("2026-07-15 23:59:59"),
}
# Secondary expanding-window folds: stability check, never used for selection.
SECONDARY_FOLDS = [
    {"name": "wf_mar", "train_end": pd.Timestamp("2026-03-15 23:59:59"),
     "val_start": pd.Timestamp("2026-03-16"), "val_end": pd.Timestamp("2026-04-14 23:59:59")},
    {"name": "wf_apr", "train_end": pd.Timestamp("2026-04-15 23:59:59"),
     "val_start": pd.Timestamp("2026-04-16"), "val_end": pd.Timestamp("2026-05-15 23:59:59")},
    {"name": "wf_may", "train_end": pd.Timestamp("2026-05-15 23:59:59"),
     "val_start": pd.Timestamp("2026-05-16"), "val_end": pd.Timestamp("2026-06-14 23:59:59")},
]
# Embargo applied around the target-encoding boundary only (longest window).
EMBARGO_S = WINDOWS_S["168h"]

# --------------------------------------------------------------------------
# Columns banned from the model matrix
# --------------------------------------------------------------------------
# transaction_id and raw epoch time are monotone in time: a tree would split on
# "after row N" and fall off a cliff at the train/test boundary.
BANNED_FEATURES = {
    ID_COL,
    TIME_COL,
    TARGET,
    "ts_epoch",
    "is_test",
    "row_order",
    CUSTOMER,
    MERCHANT,
    DEVICE,
}

FLOAT_DTYPE = np.float32

# --------------------------------------------------------------------------
# Integrity switch: signup-date inconsistency
# --------------------------------------------------------------------------
# `signup_inconsistency_d` compares a row's implied signup day against the one
# the customer's earlier rows established. Computed past-only it is a perfectly
# ordinary data-quality feature -- and in real fraud work, "this account's
# stated age contradicts its own history" is a genuine identity-tampering
# signal.
#
# In THIS dataset, however, it is near-deterministic: rows where it is non-zero
# are 99.1-100% fraud (56x lift) and it covers ~7.5% of all fraud, while only
# 0.001% of legitimate rows show any inconsistency at all. That pattern is a
# fingerprint of how the fraud rows were synthesised, not behaviour the model
# is meant to learn. The rules say:
#
#   "Reverse-engineering the generative assumptions is not the intended path to
#    a good score; understanding behavior, generally, is."
#
# and reproducibility review can flag work that leans on generation artefacts.
# It is not target leakage -- it uses only raw non-target columns, past-only --
# so this is a judgement call about the spirit of the rules, left explicit and
# switchable rather than buried. Default off.
USE_SIGNUP_INCONSISTENCY = False
