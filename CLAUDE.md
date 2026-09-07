# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Feature engineering + modelling for the REACT 2026 Kaggle datathon (IEEE SEU SB):
predict per-transaction fraud probability, scored by **PR-AUC / average
precision**. Competition rules are in `kaggle_description.md` and
`DATATHON_RULEBOOK.pdf`; `README.md` carries the measured results and findings;
`SUBMISSIONS.md` is the running decision log of what each leaderboard
submission was spent to learn.

## Environment overrides

Three settings are read from the environment so a run can be reconfigured
without editing tracked source. Defaults reproduce the local layout exactly.

```bash
REACT_SIGNUP=1        # enable the integrity feature for one build (default off)
REACT_DATA=<dir>      # where train.csv / test.csv / sample_submission.csv live
REACT_PROCESSED=<dir> # where the pipeline writes caches and results
```

`REACT_SIGNUP` exists because rewriting the literal in `src/config.py` and
restoring it afterwards left the repo on the wrong value twice when a build was
killed mid-run. `REACT_DATA` / `REACT_PROCESSED` exist for the reproducibility
notebook: on Kaggle the repository is mounted **read-only** under
`/kaggle/input`, the competition CSVs sit in a different input directory again,
and `/kaggle/working` is the only writable path -- so a hard-coded
`ROOT / "data"` fails at import, before anything can be verified.

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

# Stage 0 diagnostics -- run these before trusting any modelling number
.venv/Scripts/python.exe -u baseline.py                  # raw-column reference AP
.venv/Scripts/python.exe -u adversarial_validation.py    # train-vs-test drift, all modes
.venv/Scripts/python.exe -u adversarial_validation.py --mode iterate   # ~25 min

# Does a change help on the window that tracked the leaderboard?
.venv/Scripts/python.exe -u ablate.py --set blocks   # leave-one-block-out
.venv/Scripts/python.exe -u ablate.py --set drift    # the time-counter question
.venv/Scripts/python.exe -u ablate.py --set stage1   # new blocks + encoder delay

# Stage 2: does a change to the *fitting procedure* help? Selection on
# tail_late, confirmation on tail_recent, one fit per cutoff.
.venv/Scripts/python.exe -u tune.py --set params          # ~10 min, 5 LightGBM configs
.venv/Scripts/python.exe -u tune.py --set drift --far     # recency + window truncation
.venv/Scripts/python.exe -u tune.py --set family          # XGBoost / CatBoost
.venv/Scripts/python.exe -u tune.py --set diverse --far   # members proposed for decorrelation
.venv/Scripts/python.exe -u tune.py --set all

# Which local statistic actually predicts the board? Fits nothing; needs --far
# to have been run over every member first.
.venv/Scripts/python.exe -u horizon_proxy.py

# Stage 3: pick weights and post-processing from the persisted tail predictions.
# Fits nothing; seconds.
.venv/Scripts/python.exe -u blend.py

# Refit on all labelled data and write submissions/<name>.csv + .json.
# REACT_SIGNUP=1 enables the integrity feature for one build without editing config.
# Members are cached per (member, seeds, determinism, feature hash), so a second
# blend built from the same members costs seconds rather than another refit.
.venv/Scripts/python.exe -u submit.py --name s2_pipeline --seeds 2 --note "..."
.venv/Scripts/python.exe -u submit.py --name final --seeds 3 --deterministic
# Reproduce an earlier submission: pin the round book it was built from, because
# tune.py's early stopping is not stable across feature rebuilds.
.venv/Scripts/python.exe -u submit.py --name redo --seeds 3 --deterministic \
    --rounds-book tune_rounds.s5_asbuilt.json --weights '{"cat_base": 1.0}'

# Generate the rulebook-8.2 reproducibility notebook for a written submission.
# Reads submissions/<name>.json, so it names the exact members, weights, round
# counts and feature hash that produced the upload rather than a description of
# them, and ends by diffing its own output against the uploaded CSV.
.venv/Scripts/python.exe make_notebook.py --name s6_det_s5comp

# Superseded by tune.py/blend.py/submit.py, kept for the older numbers.
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

## Validation does not predict the leaderboard, and why

`primary_62d` reports ~0.725. The leaderboard returned **0.527**. The gap is not
a leak — it is a drift the fold averages away, and it is the most important
thing to know before trusting any local number here.

The fraud amount signature decays across the stream while legitimate behaviour
does not. Fraud median amount falls 4,918 BDT (January) -> 2,044 (May) -> 944
(mid-July); the legitimate median holds at ~485 for all 37 weeks. Raw
`amount_bdt` loses 3x of its standalone AP (0.45 -> 0.15). Inside `primary_62d`
the weekly AP is flat near 0.79 for six weeks and then falls to 0.599 and 0.473
in the final two — which average to the leaderboard score. The test period sits
entirely in the late regime and is still moving.

The *relative* amount forms do not escape it (`amt_ratio_median` 0.433 -> 0.275,
`log_amt_ratio_mean` 0.421 -> 0.257). They were built to absorb a shift in the
population distribution; this is a shift in the fraud distribution, so
normalising by the customer's own history does not help. Velocity/recency and
graph features do hold up: `cust_dt` retains 0.89 of its power, `gnn_cd_cos`
0.99.

Consequences for any new work:

- **Score candidates on the tail window (2026-06-29 -> 2026-07-15) as well as
  `primary_62d`, and believe the tail when they disagree.** It is now a real
  fold, in two geometries:
  - **`tail_late`** — cutoff 2026-05-14, the same as `primary_62d`, so it costs
    no extra fit: one model, two validation masks. 46–62 days ahead, newest
    regime. This is the selection target. `validation.selection_folds()`
    returns `[tail_late, primary_62d]` and `harness.run_folds` groups folds by
    cutoff so the shared fit happens automatically.
  - **`tail_recent`** — cutoff 2026-06-28, same rows, 1–17 days ahead. The only
    geometry that can see recency adaptation working.

  67,196 rows / 1,068 positives. Its **measured** 95% bootstrap band is
  ±0.027 AP, wider than the ±0.02 previously assumed — so a bare point estimate
  from it is not evidence. Use `evaluate.paired_ap_delta`, which scores both
  candidates on identical rows so the shared sampling noise cancels; `ablate.py`
  does this automatically and prints a verdict.

  **`tail_late` has a resolution limit of roughly 0.008 AP, measured against
  four leaderboard results.** Above it the proxy works and passes through at
  29–41% (+0.035 local → +0.0144 board; +0.025 → +0.0072). Below it the proxy
  does not merely shrink, it **loses the sign**: submission 3 gained +0.0023 on
  `tail_late` and lost 0.0025 on the board. This is stronger than a noise
  argument. A paired bootstrap can report a tight interval on a sub-0.005 delta
  and that interval is still meaningless, because 1,068 positives cannot
  separate models that close. **Do not spend a submission on a local delta below
  ~0.008, and do not adopt one either.** Everything Stage 2 measured — every
  hyperparameter, every recency half-life, every window, every model family,
  every post-processing variant — lives below that line.
- **Any conclusion measured only on `primary_62d` is suspect if it concerns
  drift.** "Recency weighting does not pay" was the live example: half-lives of
  45 d and 90 d had been rejected against a six-week old-regime average, the one
  objective that structurally cannot see drift adaptation working. It has since
  been re-measured properly on `tail_recent` and the verdict **held** — see
  "Recency weighting is settled" below.
- Report AP as lift over each slice's own base rate as well as raw AP. Weekly
  base rates range 0.0139–0.0193 and raw AP moves with them. `evaluate.Score`
  carries the base rate and lift alongside every AP so this is not optional.

### The drift is in the label mechanism, not the population

Measured by `adversarial_validation.py`, and this rules out a whole family of
fixes:

- **Train and test are near-indistinguishable on the raw covariates.**
  Adversarial AUC is 0.612 over the raw columns, and **0.514 — chance — once
  `account_age_days` is removed**, which separates only because it counts one
  per day for every returning customer. `amount_bdt` has PSI 0.0002 and KS
  0.0030 between train and test: the amount *distribution* did not move at all.
- What moved is `P(fraud | amount)` — fraud median amount 4,918 → 944 BDT while
  the legitimate median holds at ~485. That is **concept drift, not covariate
  shift.**
- Therefore **density-ratio / inverse-propensity importance weighting cannot
  help.** It corrects a shift in `p(x)` and assumes `p(y|x)` is fixed, which is
  exactly backwards here. Two independent LightGBM studies also found it losing
  to an untouched baseline (Pan et al., AdKDD 2020: IPW lost on all six drifted
  datasets by 1.7–4.5 AUC; Qian et al.: 0.7202 vs 0.7237). The weights are
  still emitted to `adversarial_weights.parquet` so the claim stays checkable,
  and they are nearly uniform (p99 = 1.83, ESS 93.9%) — there is almost nothing
  for them to reweight.
- The same papers found adversarial **feature selection** helping where
  weighting hurt. That is the branch worth pursuing, via `ablate.py --set drift`.

**PSI thresholds are unusable at this scale.** The familiar 0.10/0.25 bands are
Lewis (1994), calibrated for samples in the hundreds. PSI is asymptotically
`(1/n + 1/m)·χ²_{B-1}`, so at 67k-vs-67k the 5% critical value is ~0.0005 —
roughly 500× below the rule of thumb. Use PSI as an **effect size for ranking**
which features moved most; `adversarial_validation.psi_critical` gives the
honest pass/fail.

## Submissions are a scarce resource

**10 total, max 5 per day**, and only 2 may be selected for private scoring
(leaderboard is 60% public / 40% private). `SUBMISSIONS.md` is the ledger —
every entry records the question the submission was spent to answer, not just
its score. Update it after each result.

**The noise floor is not one number — it depends on what varied.** Measured on
the board, not assumed:

| what changed | board delta | source |
|---|---|---|
| seeds 2→3 + `deterministic=True`, same matrix, pinned rounds | **0.00019** | s5 → s6 |
| the blend weight vector only | **+0.00155** | s6 → s7 |
| feature matrix rebuilt (GNN float churn) | ~0.002, and round counts move 3× | `lgb_deep` 230 → 758 rounds |

The old blanket "~0.003" came from a comparison that *also* rebuilt the
features, so it conflated the model's variance with the matrix's. Hold the
matrix fixed and the board resolves differences an order of magnitude smaller
than this repo has been treating as unresolvable — which is what made the
weighting result (+0.00155, ~8× the measured floor) readable rather than noise.

Two rules survive intact. **The local tail still cannot resolve below ~0.008**
regardless of how precise the board is — that limit is about 1,068 positives,
not about determinism. And a *small* board delta between two genuinely different
candidates is still weak evidence, because selecting the max-public of several
near-identical files imports the public set's luck into the choice.

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

Feature blocks: `temporal` 18, `amount` 40, `velocity` 57, `entity` 55,
`behaviour` 28, `encoding` 13, `graph` 29 = **243 columns**. Gain share:
`amount` 48.6%, `velocity` 27.4%, `entity` 13.1%, `graph` 3.4%, `temporal`
2.9%, `behaviour` 2.4%, `encoding` 2.4%.

**Gain share is not marginal value.** `encoding` and `behaviour` both carry
2.4% of gain; removing the encoder's feedback delay costs 0.008–0.028 AP on
every window, while removing `behaviour` entirely costs nothing measurable.
Use `ablate.py` to decide what matters, not `train_check.py`.

`behaviour.py` tags its columns with **sub-blocks** via `out.attrs["subblock"]`
(`habit`, `repetition`, `travel`, `youth`), which propagate into the manifest
and the cache stamp as `behaviour:habit` and so on. That is what lets
`ablate.py` give each idea its own verdict instead of one lumped answer — and
they differ by an order of magnitude standalone, so the distinction earns its
keep.

**Two scoring helpers everything should route through.** `src/evaluate.py` owns
AP, lift, weekly and horizon breakdowns, bootstrap bands and the paired
bootstrap; `src/harness.py` owns fold running, which groups folds by cutoff so
`primary_62d` and `tail_late` come from one fit, and rebuilds the target
encoder per cutoff. Do not re-implement either — the reason the tail window was
easy to forget before is that reporting it was a manual step.

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

**Unbounded counters are `transaction_id` wearing a disguise, and
`BANNED_FEATURES` does not catch them.** The adversarial run separates train
from test at **AUC 1.0000** on the engineered matrix. 44 features are monotone
in stream position *and* separable on their own; the worst have test values
almost entirely outside their train range — `merch_age_days` (AUC 0.997, mean
98.7 → 226.2), `dev_age_days` (0.995), `g_comp_size` (0.981, 38,547 → 57,356),
`gnn_cm_cos` (0.933). Every test row sits past the largest split point the trees
ever saw, so the feature is effectively constant at scoring time no matter what
gain it earned in training. The iterative procedure (`--mode iterate`) had to
strip **90 features** before the AUC came near 0.75, and it plateaued at 0.79 —
the separability is pervasive, not a few bad columns.

Two consequences. Prefer **rates and shares over lifetime counts** when adding
anything new (`loc_switch_rate_24h`, not `loc_switches_lifetime`) — and note
that `*_age_frac` is not automatically safe either: `dev_age_frac` still
separates at AUC 0.678 because it saturates toward 1 as the stream lengthens.
And treat the drop list as a hypothesis: `ablate.py --set drift` measures it on
the tail, because Pan et al. also record a 12-point AUC collapse from
over-dropping.

**Two feature modules can emit the same column name.** `amount.py` and
`velocity.py` independently arrived at `cust_amt_mean_24h`. `pd.concat` then
gives a duplicated column, `features[col]` returns a DataFrame, and every
per-column check downstream crashes or silently skips. `check_no_banned_columns`
now fails on duplicates; `amount.py` imports `velocity._KEYS` so the two cannot
drift apart.

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
bound, and `leakage_checks.check_tail_coverage` now guards it — it compares
each feature's NaN rate over the last 7 days of the stream against its rate
elsewhere. When adding any snapshot/blocked feature, check the NaN rate on the
last rows of the stream, not just the aggregate.

**"Early-stop on the widest window and report the narrow ones as free reads"
was false, because the narrow window was *inside* the wide one.** `tail_late`
(06-29 → 07-15) is a strict subset of `primary_62d` (05-15 → 07-15): 67,196 of
its 240,065 rows and **1,068 of its 4,028 positives**. So stopping on
`primary_62d` picked the round count using 28% of the selection window's own
labels, and every `tail_late` number produced that way was mildly optimistic —
including the Stage 0/Stage 1 ablation table. `validation.stopping_fold` now
returns a **registered window disjoint from every tail window** (`es_late`,
05-15 → 06-28; `far_wide`, 04-16 → 06-28) and asserts the disjointness, falling
back to the widest fold only where nothing is registered. `primary_62d` still
overlaps its own stopping window by 45 of 62 days — unavoidable, and accepted
because it is no longer a selection target.

Two lessons worth keeping. A nested window is not an independent read, however
wide the outer one is; check containment, not width. And the contamination was
*shared* across candidates, so the paired ablation deltas measured before the
fix still stand — it is the absolute levels that were inflated. When a defect
like this turns up, work out which of the two it damages before rerunning
anything.

**Recency weighting is settled, and the reason is worth keeping.** Half-lives
of 7/14/21/45/90 d and hard training windows of 45/90/120 d, all measured on
`tail_recent` (the fold whose training data ends *inside* the new regime, and
the only one that can see adaptation working): **not one clears the noise
floor**. The explanation is that `win_45d` trains on 175,571 rows — 24% of the
labelled data — and scores the same as training on all 731,942. Three quarters
of the training set contributes nothing measurable, so a weighting scheme meant
to down-weight stale rows has nothing left to correct. Do not re-litigate this
without a new mechanism.

**Entity post-processing: the version that works is the version that cheats.**
Shrinking each score toward its entity's mean (the IEEE-CIS move) costs
**−0.09 to −0.28 AP** — fraud is per-transaction here, so pooling within a
customer deletes the within-entity ordering AP is made of. Shrinking toward the
entity *max* looked like a real +0.0040 with a paired interval of
[+0.0019, +0.0063] — until it was recomputed as `cummax`, which restricts each
row to its entity's *earlier* rows. The gain fell to +0.0004. The whole effect
was a July test row being adjusted by a September one. `blend.py` will not adopt
a non-causal variant regardless of its score.

**`scale_pos_weight` makes it worse here.** Measured on the raw-column
baseline over all six windows: plain 0.3167 / spw=55.8 0.3071 on `primary_62d`,
0.2259 / 0.2145 on `tail_late`, and the unweighted model wins **6 of 6**. The
metric only reads the ordering, so upweighting the positive class adds no
information and only distorts the leaf values. The research literature
recommends it over resampling, which is well supported — but "better than
SMOTE" is not "better than nothing", and on this data it is not.

**Fitted blend weights need a higher bar than the paired bootstrap gives
them.** The hill-climb optimises its weights on the same 1,068 tail positives
the bootstrap then resamples, so the interval it reports is of a quantity
already fitted to the data -- it cannot see its own selection bias. Run over 16
members it adopted a blend containing `hl_7d`, individually the *worst* member
in the pool, which is what reaching for a decorrelated member to squeeze one
window looks like. `blend.py` now picks the best **unweighted** option outright
(one model, or an equal average over an a-priori group -- neither learns
anything from the tail) and requires a fitted blend to beat that by **twice**
the noise floor. On the current pool the hill-climb beats `equal_core` by
+0.0036 and the best unweighted option by +0.0008, and is rejected.

**The old "XGBoost and CatBoost score below every LightGBM config" note is
stale.** That was measured on the 201-column set against `primary_62d`.
Re-measured on 243 features: `cat_base` is the **best single model in the
project** (tail_late 0.5530, primary 0.7375 against the 0.7239 the note was
written about) and `xgb_base` is the best of anything on `tail_recent`. Naive
equal-weighting across all three families is now a tie with LightGBM-only
(0.5579 vs 0.5580), not a loss. Do not exclude a family without re-measuring.

**No feature derived from any fraud label may be used** — not even past labels.
`test.csv` has no labels at all, so such a feature is computable on train and
structurally absent at scoring time, inflating validation and collapsing on the
leaderboard.

## Integrity flag

`config.USE_SIGNUP_INCONSISTENCY` (default `False`, set with `REACT_SIGNUP=1` in
the environment — **do not** edit the literal in `src/config.py`; a killed
process left the repo on the wrong value twice when builds worked that way). The
feature is past-only
and not target leakage, but in this dataset it is 85.8% precise — a fingerprint
of how fraud rows were synthesised rather than behaviour. The rules say
reverse-engineering the generative assumptions "is not the intended path", and
top-15 teams face reproducibility review. Worth ~+0.025 AP. Do not flip it
without asking the user; the reasoning is documented in `README.md`.

**The feature build is not bit-reproducible, and that is the dominant source of
run-to-run variation — not LightGBM.** `lgb_deep` early-stopped at 230 rounds on
one day and 758 on the next, same code and data, for a `tail_late` change of
0.0011. `lgb_base` meanwhile reproduced to four decimals across three separate
processes in one session. The cause is `gnn.py`: it is correctly seeded, but
multi-threaded CPU reductions in torch are not deterministic, so the four
`gnn_*` columns rebuild to ~3e-4 (correlation 1.000000) rather than exactly.
That is enough to move an early stop by 3×, which means the AP stopping curve is
a plateau, not a peak. Three rules follow:

- **Pin round counts when reproducing a submission.** `submit.py --rounds-book`
  takes the book it was built from; `tune_rounds.s5_asbuilt.json` holds the
  counts behind 0.55014. Naming the same members does *not* identify the same
  model.
- **Ship the matrix, not just the code.** A rerun ranks the test set essentially
  identically and scores within the noise floor, but the CSV is byte-identical
  only if the built matrix travels with the notebook.
  `data/processed/base_features.signup.parquet` preserves the 244-column build.
- **A paired-bootstrap verdict does not survive a rebuild.** `lgb_shallow` read
  −0.0007 against `lgb_base` on one build and +0.0032 / `BETTER` / interval
  excluding zero on the next. The verdict machinery will promote a candidate on
  early-stopping churn alone.

**Add ensemble members for decorrelation, and measure it with rank ρ — not
family labels and not head overlap.** Calibrated against the two board results:
`cat_base` entered at ρ 0.624 and gained +0.0014; eight recency/window variants
entered at ρ 0.736–0.826 and cost 0.0025; core members sit at 0.840–0.915 among
themselves. Top-1% overlap does *not* separate the winner from the losers (0.934
vs 0.931) while ρ does, cleanly. `xgb_base` sits at ρ 0.824 — inside the clone
band — so a different library is not automatically a different model.
`blend.py` prints this table.

**Decorrelation and competence trade off on this data.** The two most
independent members ever built here (`lgb_linear` ρ 0.572, `cat_durable` 0.632)
are exactly the two that collapse at `tail_far`; the one that matches the core
at every horizon (`lgb_sub30`) is a clone at 0.829. `cat_base` is the only
member that is both, which is why it is the only addition that ever paid. Do not
assume a replacement can be found by trying more variants — four were tried with
stated mechanisms and all four failed.

**Score `tail_far` before adopting anything.** `lgb_linear` ties the core on the
selection target, clears the ρ gate, and has a documented mechanism — it would
have passed every check that existed before. At 75–91 days it collapses from
0.5484 to 0.5016. Blend decay also rises with member count (0.0169 at 5 members,
0.0197 at 13, 0.0190 at 20), which is the mechanism behind submission 3's board
loss: extra members buy near-horizon AP and pay for it in the regime the private
40% sits in.

**`mean(tail_late, tail_far)` is a better level estimator and still cannot
discriminate.** Board-vs-local error falls from +0.0096/+0.0141 to
+0.0012/+0.0043, but it still gets the s2-vs-s3 ordering wrong. The 0.008
resolution limit is therefore not an artefact of a badly-centred statistic —
1,068 positives cannot resolve 0.0025 whatever functional is computed from them.
`horizon_proxy.py` runs the test.

## Caching

`data/processed/base_features.parquet` caches the label-free matrix (~2 min to
rebuild, and `graph.build` is ~90s of it). `harness.load_base` is the single
entry point (`train_model.load_base` delegates to it) and it keys the cache on a
**SHA-256 of every feature module's source plus the config values that change
feature values**, written to `base_features.stamp.json`. Editing a feature
module now invalidates the cache automatically; the old manual "delete it or
you will train on stale features" step is gone. Deleting the parquet by hand
still works and is harmless.

## Reference numbers

`baseline.py` gives the denominator everything else is measured against — a
bare LightGBM on raw columns only (amount, account age, hour, day of week, the
five low-cardinality categoricals):

| window | raw baseline AP | lift |
|---|---|---|
| `tail_late` | 0.2259 | 14.2× |
| `tail_recent` | 0.2293 | 14.4× |
| `primary_62d` | 0.3167 | 18.9× |
| `wf_mar` / `wf_apr` / `wf_may` | 0.4238 / 0.3714 / 0.3492 | ~21× |

Gain share within the raw model: `amount_bdt` 59.7%, `hour` 17.6%, `location`
9.1%, `account_age_days` 5.6%. Retention from early to tail: `amount_bdt` 0.64,
`hour` 0.68, `account_age_days` **1.00**, categoricals ~0.93.

## Known open items

- `te_merchant_id` ranks 8th by gain but was never A/B'd against dropping it.
  `ablate.py --set blocks` can now drop the whole `encoding` block; a
  single-column variant still needs adding.
- Final-model round counts are fold counts scaled by `sqrt(n_final / n_ref)`
  (= 1.22x), which is the same damped rule `harness.run_folds` uses for a solo
  cutoff. It replaces a hard-coded 1.2 and happens to land on the same number.
- `deterministic=True` is off; `submit.py --deterministic` enables it (with
  `force_row_wise` and a fixed thread count) at roughly 2x the fit time. Use it
  for anything that could be selected for private scoring — rulebook 8.2
  forfeits a position that cannot be reproduced from the submitted notebook.
- **The far-horizon reads are incomplete.** `tune.py --far` scores `tail_far`
  (75–91 days ahead); the drift set has it, the params and family sets do not,
  so the chosen blend has no `tail_far` number. Worth filling: the leaderboard
  landed between `tail_late` and `tail_far` and close to their mean, and if that
  holds it is the proxy the final two submissions should be selected on.
- The 44 unbounded time-counters have a measured drop verdict from
  `ablate.py --set drift`; replacing them with rate/share forms rather than
  simply deleting them is not done.

## Other agent configs

User-level OpenAI Codex and Gemini CLI configs exist on this machine. To import
them, reply `/import` to scan and list what is importable, then
`/import --yes=<digest>` to apply. If `/import` is unavailable here, run
`claude import` from a terminal.
