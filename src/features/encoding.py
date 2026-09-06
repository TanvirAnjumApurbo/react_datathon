"""Past-only target encoding. The highest leakage-risk block in the pipeline.

Two rules bind here at once:

  * "using a transaction's own fraud label, or any transaction's future fraud
     label, to construct a feature" is target leakage;
  * `test.csv` has no labels at all.

That second point is the awkward one. A train row can accumulate a growing
label history, but a test row can never have one. If we ignore that, the
encoder means something different on each side of the boundary and validation
becomes a fantasy.

The resolution used here: the encoder is **frozen at the train cutoff**. Every
row scored after the cutoff -- whether a real test row or a validation row in a
backtest -- reads the same frozen table. `fit_cutoff` makes that explicit, so a
backtest fold reproduces the exact asymmetry the leaderboard will impose.

Deliberately excluded: any feature derived from a customer's own past labels
(e.g. "this customer has been defrauded before"). It measures 2.0x lift on
train (2.79% vs 1.38%) and is computable there, but is structurally absent for
test rows. It would inflate validation and collapse on the leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

# Columns worth encoding. location is the strongest candidate (a 2.9x spread
# from LOC_007 at 5.15% down to LOC_017 at 1.18%); merchant_id is the weakest
# (per-merchant fraud-rate std is only 0.0088) and is on probation.
TE_COLS = [
    "location",
    "merchant_category",
    "device_type",
    "payment_method",
    "transaction_type",
]
TE_CROSSES = [
    ("hour_bucket", "device_type"),
    ("location", "transaction_type"),
]


def _hour_bucket(ts: pd.Series) -> np.ndarray:
    """Coarse time-of-day bands: night is where the signal lives."""
    h = ts.dt.hour.to_numpy()
    return np.select(
        [h < 5, h < 12, h < 18, h < 22], [0, 1, 2, 3], default=4
    ).astype(np.int64)


def _expanding_te(
    codes: np.ndarray,
    y: np.ndarray,
    trainable: np.ndarray,
    alpha: float,
    prior: float,
) -> np.ndarray:
    """Strictly-past expanding smoothed mean of the target, per category.

    `trainable` marks rows whose label may be consumed (train rows before the
    cutoff). Rows outside it contribute nothing to the running sums, so the
    encoder naturally freezes once the labelled period ends.
    """
    contrib_y = np.where(trainable, np.nan_to_num(y, nan=0.0), 0.0)
    contrib_n = trainable.astype(np.float64)

    s = pd.Series(contrib_y)
    n = pd.Series(contrib_n)
    g = pd.Series(codes)
    # cumsum includes the current row; subtract it to stay strictly past.
    cum_y = s.groupby(g, sort=False).cumsum().to_numpy() - contrib_y
    cum_n = n.groupby(g, sort=False).cumsum().to_numpy() - contrib_n
    return (cum_y + prior * alpha) / (cum_n + alpha)


def _decayed_te(
    codes: np.ndarray,
    y: np.ndarray,
    trainable: np.ndarray,
    epoch: np.ndarray,
    halflife_d: float,
    alpha: float,
    prior: float,
) -> np.ndarray:
    """Exponentially time-decayed variant, so the encoder tracks drift.

    Maintained as a running (weighted_sum, weight) pair per category, decayed
    to the current row's timestamp. One pass, no per-row lookback.
    """
    lam = np.log(2.0) / (halflife_d * 86_400.0)
    n = len(codes)
    k = int(codes.max()) + 1
    acc_y = np.zeros(k)
    acc_n = np.zeros(k)
    last_t = np.zeros(k)
    seen = np.zeros(k, dtype=bool)
    out = np.empty(n)

    y_f = np.nan_to_num(y, nan=0.0)
    for i in range(n):
        c = codes[i]
        t = epoch[i]
        if seen[c]:
            d = np.exp(-lam * (t - last_t[c]))
            acc_y[c] *= d
            acc_n[c] *= d
        else:
            seen[c] = True
        last_t[c] = t
        out[i] = (acc_y[c] + prior * alpha) / (acc_n[c] + alpha)
        if trainable[i]:
            acc_y[c] += y_f[i]
            acc_n[c] += 1.0
    return out


def build(df: pd.DataFrame, fit_cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build target-encoded columns.

    Parameters
    ----------
    fit_cutoff
        Labels at or before this timestamp may be consumed; everything after
        reads a frozen encoder. Defaults to the end of the labelled train
        period. **Backtests must pass their own fold cutoff**, otherwise the
        validation block sees an encoder that the real test rows will not have.
    """
    if fit_cutoff is None:
        fit_cutoff = C.TRAIN_END

    ts = df[C.TIME_COL]
    epoch = df["ts_epoch"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=np.float64)
    trainable = (~df["is_test"].to_numpy()) & (ts <= fit_cutoff).to_numpy()
    prior = float(np.nanmean(y[trainable])) if trainable.any() else 0.0

    out = pd.DataFrame(index=df.index)
    out.attrs["fit_cutoff"] = str(fit_cutoff)

    frames = {c: pd.factorize(df[c], sort=True)[0].astype(np.int64) for c in TE_COLS}
    frames["hour_bucket"] = _hour_bucket(ts)

    for col in TE_COLS:
        codes = frames[col]
        out[f"te_{col}"] = _expanding_te(codes, y, trainable, C.TE_ALPHA, prior)
        out[f"te_{col}_decay"] = _decayed_te(
            codes, y, trainable, epoch, C.TE_HALFLIFE_D, C.TE_ALPHA, prior
        )

    for a, b in TE_CROSSES:
        cross = pd.factorize(frames[a] * 1000 + frames[b], sort=True)[0].astype(np.int64)
        out[f"te_{a}_X_{b}"] = _expanding_te(cross, y, trainable, C.TE_ALPHA, prior)

    # merchant_id: high cardinality (4,290) and weak (fraud-rate std 0.0088).
    # Kept with heavy smoothing so the model can reject it on gain.
    m_codes = df[f"{C.MERCHANT}_code"].to_numpy().astype(np.int64)
    out["te_merchant_id"] = _expanding_te(m_codes, y, trainable, C.TE_ALPHA * 4, prior)

    float_cols = [c for c in out.columns if out[c].dtype == np.float64]
    return out.astype({c: C.FLOAT_DTYPE for c in float_cols})
