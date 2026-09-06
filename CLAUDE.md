# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Feature engineering + modelling for the REACT 2026 Kaggle datathon (IEEE SEU SB):
predict per-transaction fraud probability, scored by **PR-AUC / average
precision**. Competition rules are in `kaggle_description.md` and
`DATATHON_RULEBOOK.pdf`; `README.md` carries the measured results and findings.

## Interpreter

**Always use `.venv/Scripts/python.exe`.** Bare `python` on PATH resolves to
msys2 mingw Python, which has no pip and none of the dependencies. Neither of
the two system Pythons has a complete stack either (3.10 has pandas but no
scipy/sklearn; 3.12 has scipy/sklearn but no pandas).

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Commands

```bash
# Features: build everything, prove past-only, write parquets + manifest
.venv/Scripts/python.exe build_features.py --check
.venv/Scripts/python.exe build_features.py --no-gnn      # skip Tier C, ~90s faster

# Feature validation across walk-forward folds (LightGBM gain + fold AP)
.venv/Scripts/python.exe train_check.py

# Modelling. Stages are independently runnable and cache their outputs.
.venv/Scripts/python.exe -u train_model.py --stage search   # ~15 min, 7 configs
.venv/Scripts/python.exe -u train_model.py --stage blend    # seconds, reads val_preds
.venv/Scripts/python.exe -u train_model.py --stage final    # ~45 min, 15 models
.venv/Scripts/python.exe -u train_model.py --stage all

# Rescale submission to probability range (ordering/PR-AUC unchanged)
.venv/Scripts/python.exe calibrate_submission.py [--inplace]
```

Use `-u` for the long runs — output is block-buffered when redirected and you
will otherwise see nothing for 15+ minutes.

There is no test suite. **`build_features.py --check` is the test**: it runs the
leakage assertions in `src/leakage_checks.py` and fails loudly. To run one check
in isolation, import it directly:

```bash
.venv/Scripts/python.exe -c "
from src.io_utils import get_stream
from src.features import temporal
from src import leakage_checks as LC
import pandas as pd
LC.check_truncation_invariance(temporal.build, get_stream(), pd.Timestamp('2026-05-01'))"
```

## The three rules everything is built around

From `kaggle_description.md` — violating any of these is a rule violation,
"whether accidental or deliberate":

1. **Strictly past-only.** A feature for a transaction at time *t* may use only
   information from before *t*.
2. **Combined-stream features.** Features are computed over
   `concat(train, test)` sorted by `(timestamp, transaction_id)`. This is
   explicitly permitted *and required* — building on train alone starves every
   test row of its within-test history and makes test features systematically
   unlike train features.
3. **No random K-fold.** Forward-block validation only.

## Architecture

The stream is the spine. `io_utils.get_stream()` returns one time-sorted frame
of train+test with `is_test` and `fraud` (NaN for test). Every feature module
takes that frame and returns only its new columns; `build_features.py`
concatenates them.

**`src/features/_windows.py:WindowIndex` is the single chokepoint for the
past-only guarantee.** It resolves each row's window start with one global
`searchsorted` over a composite `group_code * 2**31 + ts_epoch` key. Route new
historical features through it rather than reimplementing lookback — the
guarantee then lives in one place instead of being re-argued per feature.

Feature blocks and their measured gain share: `amount` 47%, `velocity` 27%,
`entity` 12%, `encoding` 7%, `graph` 4%, `temporal` 3%.

`graph.py` uses **snapshot-and-join**: at each weekly boundary it builds the
graph from strictly-earlier rows, computes node quantities, and joins them onto
the following week's transactions. It exports only **rotation-invariant**
quantities (cosine similarities, drift distances) — raw SVD coordinates are
sign- and rotation-arbitrary across snapshots and behave as noise.

## Traps that have already bitten this codebase

**`encoding.build(df, fit_cutoff=...)` must get the *fold's* cutoff in a
backtest.** It defaults to `TRAIN_END`, which is correct for scoring the real
test set but leaks validation labels inside a fold. `train_check.py` and
`train_model.py` rebuild it per fold; anything new that backtests must too.

**Anything derived per customer belongs in a feature module, not `io_utils`.**
The truncation test calls `build_fn(df)`, so a customer-level summary computed
inside `get_stream()` runs upstream of it and escapes the check entirely. That
is exactly how a `groupby.transform("median")` spanning each customer's full
history (including future rows) survived a passing test suite.

**Do not write `timestamp.astype("int64") // 10**9`.** pandas 3 stores this
column as `datetime64[us]`, so that idiom yields *kiloseconds* and silently
rescales every window by 1000× — a "1h" window becomes 41.7 days. Use
`.astype("datetime64[s]").astype("int64")`. `validate_stream` asserts the span.

**Never add `transaction_id` or raw epoch time to the model matrix.** Both are
monotone in time, so a tree splits on "after row N ⇒ …" and extrapolates off a
cliff at the boundary. `config.BANNED_FEATURES` enforces this. The same logic
applies to anything monotone in calendar time: `dev_age_days` and
`merch_age_days` each ship with a drift-stable `*_age_frac` counterpart.

**A feature block that is complete on train can still be empty at the end of
test.** `graph.py` joins weekly snapshots onto `[bounds[i], bounds[i+1])`, and
`pd.date_range` stops at the last anchor <= end -- so the trailing partial week
got no snapshot and all 29 graph columns stayed NaN. Because the stream ends
inside the *test* period, this stranded 10,455 test rows (3.98%) and **zero
train rows**: the truncation test passed, the boundary check passed, every fold
scored normally, and the defect was still there. Fixed by appending a terminal
bound. When adding any snapshot/blocked feature, check the NaN rate on the last
rows of the stream, not just the aggregate.

**Do not equal-weight ensemble across model families.** XGBoost and CatBoost
both score below every LightGBM config here, so naive rank-averaging lands
*below* the best single model. `train_model.py:hill_climb` picks weights from
evidence; it excluded both non-LightGBM families outright.

**No feature derived from any fraud label may be used** — not even past labels.
`test.csv` has no labels at all, so such a feature is computable on train and
structurally absent at scoring time, inflating validation and collapsing on the
leaderboard.

## Integrity flag

`config.USE_SIGNUP_INCONSISTENCY` (default `False`). The feature is past-only
and not target leakage, but in this dataset it is 85.8% precise — a fingerprint
of how fraud rows were synthesised rather than behaviour. The rules say
reverse-engineering the generative assumptions "is not the intended path", and
top-15 teams face reproducibility review. Worth ~+0.025 AP. Do not flip it
without asking the user; the reasoning is documented in `README.md`.

## Caching

`data/processed/base_features.parquet` caches the label-free matrix (~2 min to
rebuild, and `graph.build` is ~90s of it). `train_model.py:load_base` reuses it
whenever the row count matches — **delete it after changing any feature module**,
or you will train on stale features.

## Known open items

- `te_merchant_id` ranks 8th by gain but was never A/B'd against dropping it;
  the plan called for gating it on fold AP and that gate is untested.
- Final-model round counts are fold counts scaled by 1.2 — a convention, not a
  validated choice.

## Other agent configs

User-level OpenAI Codex and Gemini CLI configs exist on this machine. To import
them, reply `/import` to scan and list what is importable, then
`/import --yes=<digest>` to apply. If `/import` is unavailable here, run
`claude import` from a terminal.
