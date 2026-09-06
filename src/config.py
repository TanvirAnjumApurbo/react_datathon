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
# Long RFM windows, in seconds. The Whitrow set above tops out at 7 days, but
# the ULB handbook's spending-behaviour features use [1, 7, 30] days and a
# 30-day baseline is what makes "this amount is unlike the customer's month"
# expressible. Kept separate from WINDOWS_S so adding 30d costs a handful of
# amount/velocity columns rather than multiplying every windowed feature.
RFM_WINDOWS_S = {
    "24h": 86_400,
    "168h": 604_800,
    "30d": 2_592_000,
}
# Trailing windows used for the drift-robust amount percentile ranks.
AMOUNT_RANK_WINDOWS_D = [7, 30]

# Bayesian smoothing strength for past-only target encoding.
TE_ALPHA = 100.0
# Half-life (days) for the time-decayed target-encoding variant.
TE_HALFLIFE_D = 30.0
# Feedback delay, in days, for every target-derived statistic.
#
# A fraud label does not exist the instant the transaction happens -- it exists
# once an investigation or a chargeback confirms it. The ULB handbook models
# this with a delay period and computes the risk for day N from labels up to
# day N-delay only. Two reasons it matters here:
#
#   1. Realism/compliance. Without it the encoder consumes a label the moment
#      the row lands, which no deployed system could do.
#   2. Honesty about drift. A zero-delay encoder is at its sharpest exactly at
#      the fold boundary and then decays; the delay makes the encoder the
#      validation rows see resemble the (much staler) one the test rows get.
#
# Set to 0.0 to restore the previous behaviour.
TE_FEEDBACK_DELAY_D = 7.0

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
# ---------------------------------------------------------------------------
# The tail folds. THESE ARE THE SELECTION TARGETS, not `primary_62d`.
#
# Submission 1 measured local 0.7252 on `primary_62d` and 0.52708 on the public
# leaderboard. The gap is drift, not leakage: the fraud amount signature decays
# across the stream while legitimate behaviour does not, so the last two
# labelled weeks score ~0.53 while the six-week fold average hides it.
#
# Two geometries matter, and they answer different questions:
#
#   tail_late    train to 2026-05-14, score 2026-06-29 -> 07-15.
#                A 46-62 day-ahead forecast into the newest regime -- the same
#                slice of `primary_62d` that scored 0.516 against an LB of
#                0.527. Horizon AND regime both match the far end of the real
#                test window, so this is the primary selection target. It
#                shares `primary_62d`'s cutoff, so scoring it is free: it is a
#                different val mask over the same trained model.
#
#   tail_recent  train to 2026-06-28, score 2026-06-29 -> 07-15.
#                A 1-17 day-ahead forecast over the same rows. Regime matches,
#                horizon does not. This is the fold that can actually see
#                recency adaptation working, because its training data ends
#                inside the new regime -- the question `primary_62d`
#                structurally cannot answer.
#
# Both windows are small (~67k rows, ~1,050 positives), so the noise band is
# wide, roughly +/-0.02 AP. Report lift over the slice's own base rate
# alongside raw AP: weekly base rates range 0.0139-0.0193 and raw AP moves
# with them.
# ---------------------------------------------------------------------------
TAIL_START = pd.Timestamp("2026-06-29")
TAIL_END = pd.Timestamp("2026-07-15 23:59:59")

TAIL_LATE_FOLD = {
    "name": "tail_late",
    "train_end": pd.Timestamp("2026-05-14 23:59:59"),
    "val_start": TAIL_START,
    "val_end": TAIL_END,
}
TAIL_RECENT_FOLD = {
    "name": "tail_recent",
    "train_end": pd.Timestamp("2026-06-28 23:59:59"),
    "val_start": TAIL_START,
    "val_end": TAIL_END,
}

# ---------------------------------------------------------------------------
# A deliberate stress fold, one step beyond what the real task asks.
#
# The test window runs 1-62 days past its cutoff, which `tail_late` matches at
# its far end. But the fraud amount signature is still moving at 2026-07-15:
# fraud median amount has fallen 4,918 -> 2,044 -> 944 BDT while the legitimate
# median holds near 485. Extrapolate that and by September the two medians meet,
# and `amount_bdt` -- 48.6% of model gain -- stops separating anything.
#
# `tail_far` scores the same 06-29 -> 07-15 rows from a 2026-04-15 cutoff:
# 75-91 days ahead, past anything the real test requires. It is not a selection
# target; it is a tie-breaker. Two configs that look equal on `tail_late` are
# not equal if one of them holds up here and the other falls over, because the
# far half of the test window is the part `tail_late` measures least well.
#
# `far_wide` exists only to early-stop on, so `tail_far` stays a read on a model
# that was not tuned to it.
# ---------------------------------------------------------------------------
FAR_WIDE_FOLD = {
    "name": "far_wide",
    "train_end": pd.Timestamp("2026-04-15 23:59:59"),
    "val_start": pd.Timestamp("2026-04-16"),
    "val_end": TAIL_START - pd.Timedelta(seconds=1),
}

# ---------------------------------------------------------------------------
# Early-stopping windows, and why they are separate objects.
#
# The convention was "early-stop on the widest window at this cutoff, report the
# narrow ones as free reads on a model that was not tuned to them". That is only
# true when the narrow window is *outside* the wide one, and here it was inside:
# `tail_late` (06-29 -> 07-15) is a subset of `primary_62d` (05-15 -> 07-15) --
# 67,196 of its 240,065 rows and 1,068 of its 4,028 positives. Stopping on
# `primary_62d` therefore chose the round count partly from the labels of the
# very window being reported, and `tail_late` was not the clean read the
# docstrings claimed.
#
# These folds end the day before `TAIL_START`, so they are disjoint from every
# tail window. `primary_62d` still overlaps its own stopping window by 45 of 62
# days, which is accepted deliberately: it is no longer a selection target, and
# no window at that cutoff can early-stop a 62-day fold without touching it.
# The selection target is the one that has to be clean.
# ---------------------------------------------------------------------------
# There is deliberately no equivalent for the 2026-06-28 cutoff: the stream
# ends on 07-15, so every row after that cutoff *is* the tail window. That is
# why `tail_recent` takes a borrowed round count instead of early-stopping --
# no uncontaminated choice exists there, and inventing one would be pretending.
ES_LATE_FOLD = {
    "name": "es_late",
    "train_end": pd.Timestamp("2026-05-14 23:59:59"),
    "val_start": pd.Timestamp("2026-05-15"),
    "val_end": TAIL_START - pd.Timedelta(seconds=1),
}
TAIL_FAR_FOLD = {
    "name": "tail_far",
    "train_end": pd.Timestamp("2026-04-15 23:59:59"),
    "val_start": TAIL_START,
    "val_end": TAIL_END,
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
