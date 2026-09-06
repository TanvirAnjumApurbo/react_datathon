# REACT 2026 Datathon — Leakage-Safe Preprocessing Pipeline

Feature engineering for the IEEE SEU SB REACT 2026 temporal fraud-detection
challenge. Metric: **PR-AUC (average precision)**.

The organisers ship only raw transaction fields and state that constructing
behavioural features *without looking into the future* is the competition. So
this repo is the preprocessing layer, not a model. The LightGBM run included
here exists only to confirm the split behaves and to rank the feature blocks.

## Leaderboard status

| submission | local `primary_62d` AP | public LB |
|---|---|---|
| 1 (2026-09-06) | 0.7252 | **0.52708** |

The gap is understood and is not a leak: the fraud amount signature decays
across the stream (fraud median 4,918 BDT in January, 944 in mid-July, while the
legitimate median holds at ~485), so the last two labelled weeks score ~0.53
locally -- matching the leaderboard -- while the six-week fold average hides it.
Full analysis, per-week numbers and the plan for the remaining submissions are
in **[SUBMISSIONS.md](SUBMISSIONS.md)**, which is the running decision log.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

python build_features.py --check     # build everything + prove past-only
python train_check.py                # walk-forward feature validation
python train_model.py --stage all    # config search -> blend -> submission.csv
```

Outputs land in `data/processed/`: `train_features.parquet`,
`test_features.parquet`, `feature_manifest.json`. The submission is written to
`submission.csv` at the repo root.

## Results

201 features. LightGBM, expanding-window folds, target encoder rebuilt per fold:

| fold | horizon | val AP | vs base rate |
|---|---|---|---|
| `primary_62d` | 62 d | 0.717 | 42.7× |
| `wf_mar` | 30 d | 0.803 | 42.0× |
| `wf_apr` | 30 d | 0.761 | 43.9× |
| `wf_may` | 30 d | 0.778 | 46.7× |

mean **0.765**, std 0.036. `primary_62d` is the fold to trust — it is the only
one that reproduces the real train→test geometry (a 62-day forward block
starting the day after the cutoff). The others measure stability.

Gain by block: amount 47.2%, velocity 26.6%, entity 12.5%, encoding 6.8%,
graph 3.7%, temporal 3.2%.

## Modelling

Config search on `primary_62d` (`train_model.py --stage search`):

| config | val AP | vs base | rounds |
|---|---|---|---|
| `lgb_deep` | 0.7239 | 43.1× | 1360 |
| `lgb_hl90` (recency ½-life 90 d) | 0.7222 | 43.0× | 618 |
| `lgb_base` | 0.7217 | 43.0× | 808 |
| `lgb_hl45` (recency ½-life 45 d) | 0.7216 | 43.0× | 479 |
| `lgb_shallow` | 0.7191 | 42.9× | 1283 |
| `xgb_base` | 0.7123 | 42.5× | 233 |
| `cat_base` | 0.7088 | 42.2× | 731 |

### Blending

| strategy | val AP |
|---|---|
| equal-weight, all three families | 0.7214 |
| best single (`lgb_deep`) | 0.7239 |
| equal-weight, LightGBM only | 0.7248 |
| **greedy hill-climb** | **0.7252** |

Final weights: `lgb_deep` 0.459, `lgb_hl90` 0.189, `lgb_hl45` 0.189,
`lgb_base` 0.108, `lgb_shallow` 0.054 — **XGBoost and CatBoost were excluded
entirely**, each averaged over 3 seeds.

Two things worth knowing:

- **Equal-weight cross-family blending makes it worse.** Rank-averaging the
  best LightGBM, XGBoost and CatBoost lands *below* the best single model,
  because both non-LightGBM families sit clearly lower and drag the average
  down. Blend weights are therefore chosen by greedy hill-climb (Caruana-style,
  with replacement) on the primary fold, so a member that hurts simply never
  gets selected — which is exactly what happened to xgb and cat. `--stage blend`
  prints the comparison against best-single and equal-weight baselines before
  committing, so the choice is visible rather than assumed.
- **Recency weighting does not pay.** Half-lives of 90 d and 45 d land within
  0.0006 AP of the unweighted model, so the drift in this data is not the kind
  that down-weighting old rows fixes. It does cut the round count roughly in
  half, which is a speed argument, not an accuracy one.

  > **Superseded by submission 1.** This was measured on `primary_62d`, whose
  > 62-day window is six weeks of one fraud-amount regime plus two weeks of the
  > next -- an average that cannot see drift adaptation working. Re-measure
  > against the 2026-06-29 -> 07-15 tail window before relying on it. See
  > [SUBMISSIONS.md](SUBMISSIONS.md).

The spread across all five LightGBM configs is 0.005 AP — within single-fold
noise. That argues for averaging several configs and seeds rather than trusting
the single best, which is what the hill-climb produced. The headline gain over
the best single model is only +0.0013 AP; the real reason to prefer the blend is
variance, not the point estimate, since the private leaderboard is a different
40% of the test set.

Predictions are blended as **ranks**, not probabilities: average precision
depends only on ordering, so the models' differing output scales are
irrelevant, and for the same reason no probability calibration step can affect
the score.

### Submission

`submission.csv` — 262,648 rows, ids in `sample_submission.csv` order,
values in [0.000037, 0.999956] with mean 0.0119 (against a ~1.7% base rate).

Rank-blending leaves the raw output uniform on [0, 1] with a mean of 0.50, which
is valid but does not read as a probability. `calibrate_submission.py`
quantile-maps those ranks onto the distribution of blended probabilities
measured on the validation fold. The map is strictly monotone, so the ordering —
and the PR-AUC — is unchanged; it asserts this rather than assuming it. The
pre-mapping version is kept as `submission_ranks.csv`; the two score identically
by construction.

One implementation note: the fold has fewer rows than the test set, so
nearest-index mapping collapses distinct ranks onto equal values, and those ties
*do* move the precision–recall curve. The mapping interpolates and adds a
strictly increasing epsilon to guarantee the ordering survives exactly.

## The three rules the design is built around

1. **Strictly past-only.** Every feature for a transaction at time *t* uses only
   information from before *t*.
2. **Combined-stream features.** Features are computed over `concat(train, test)`
   sorted by `(timestamp, transaction_id)`. The rules explicitly permit this —
   *"a device's known transaction history can include test-period rows that
   occurred earlier than the row being scored"* — and it is **required**:
   building on train alone would starve every test row of its within-test
   history, making test features systematically unlike train features.
3. **No random K-fold.** Forward-block validation only.

### What is deliberately excluded

- **`transaction_id` and raw epoch time.** Both are monotone in time, so a tree
  would split on "after row N ⇒ …" and extrapolate off a cliff at the boundary.
  Time enters only as cyclical or relative quantities.
- **Any feature built from past fraud labels** (e.g. "this customer has been
  defrauded before"). It measures 2.0× lift on train and is computable there,
  but `test.csv` has no labels at all, so it would be structurally absent at
  scoring time — inflating validation and collapsing on the leaderboard.
- **`signup_inconsistency_d`** — off by default. See the integrity note below.

## Layout

```
src/
  config.py              paths, windows, fold definitions, banned columns
  io_utils.py            load, clean, build the combined time-sorted stream
  validation.py          forward-block folds + average precision
  leakage_checks.py      the guards described below
  features/
    _windows.py          the past-only window primitive everything routes through
    temporal.py          hour/day, cyclical, night, per-customer circular rhythm
    amount.py            amount vs own history; drift-robust trailing ranks
    velocity.py          customer/device/merchant recency + rolling windows
    entity.py            pair novelty, device sharing, diversity, movement
    encoding.py          past-only target encoding, frozen at the train cutoff
    graph.py             snapshot bipartite graph: degrees, components, spectral
    gnn.py               self-supervised temporal GNN embeddings (Tier C)
build_features.py        orchestrator
train_check.py           walk-forward feature validation harness
notebooks/               thin Kaggle-ready wrapper
```

## How past-only is enforced

Two mechanisms, one structural and one empirical.

**Structural.** Every historical statistic routes through
`features/_windows.py:WindowIndex`, which resolves each row's window start with
a single global `searchsorted` over a composite `group * BIG + timestamp` key.
The guarantee lives in one place instead of being re-argued per feature.

**Empirical — the truncation test.** If a feature at time *t* depended on
anything after *t*, deleting the future would change it. `check_truncation_
invariance` recomputes the entire pipeline on a stream truncated at a cutoff and
diffs against the full-stream values on the shared rows. All label-free
features pass.

That test has one requirement worth stating: **anything derived per customer
must live in a feature module, not in the loader.** A customer-level summary
computed inside `get_stream()` runs upstream of the function under test and
escapes it — which is exactly how one leak survived here before being caught.

**Coverage — the check the other two could not make.** `graph.py` assigns
snapshot values to `[boundary_i, boundary_{i+1})`, and `pd.date_range` stops at
the last weekly anchor at or before the end of the stream. The trailing partial
week therefore got no snapshot, and all 29 graph columns stayed NaN for
**10,455 test rows (3.98%) and zero train rows**. Every existing guard passed:
the affected rows carry no labels, sit in no validation fold, and the truncation
test rebuilds on a stream cut mid-train, so it has a different tail entirely.
`check_tail_coverage` now compares each feature's NaN rate over the last 7 days
of the stream against its rate elsewhere and fails above a 0.10 excess —
legitimate variation measures 0.005, and a single orphaned day inside a
seven-day tail would read 0.14.

Additional guards in `leakage_checks.py`: no banned columns; no single feature
individually near-perfect (>0.85 AP is a leak, not a discovery — the best honest
one is `amt_ratio_median` at 0.44); train/test boundary continuity; history
features null on an entity's first-ever row.

### Run-to-run variation

LightGBM is seeded (`seed`, `bagging_seed`, `feature_fraction_seed`) but runs
with `num_threads=0` and without `deterministic=True`, so histogram accumulation
order varies between threads and tiny float differences compound over 1,600
boosting rounds. Two runs of identical code on identical data are therefore not
bit-identical.

It does not matter for the metric, because the churn sits where the metric does
not look. Comparing two such runs on the 252,193 rows whose features were
unchanged:

| region | agreement |
|---|---|
| top 500 | 98.4% overlap |
| top 2,626 (1%) | 99.1% overlap |
| top 1% | Spearman 0.9989 |
| bottom 90% | Spearman 0.9595 |

Average precision integrates over the head of the ranking, which reproduces at
98–99%. The 5% mean percentile shift comes almost entirely from the bottom 90%,
where every prediction sits near 0.0005 and rows are effectively tied.

The practical consequence: **treat leaderboard differences below roughly 0.003
AP as noise**, not as evidence that one configuration beats another. Setting
`deterministic=True` would make runs exactly repeatable at a real speed cost —
worth doing for a final submission that may face reproducibility review, not for
intermediate experiments.

## Findings that shaped the design

- **The two dominant signals are relative, not absolute.** `amount > 10×` the
  customer's own past mean is 92.8% fraud (52.7× lift); a customer's previous
  transaction under a minute ago is 86.0% fraud (48.9× lift). Device velocity is
  just as sharp (a <10s device gap is 83.0% fraud) while *merchant* velocity is
  worthless (0.69× lift) — the merchants are high-traffic aggregators, with the
  top 10 carrying 57.8% of all rows.
- **The fraud amount archetype drifts into the test period.** Train P99.9 is
  42,007 BDT against 24,342 in test, and P(amount > 10,000) falls from 0.69% to
  0.45%, while the night-hour share barely moves. Raw amount is the strongest
  single feature *and* the most drift-exposed, so it is always paired with
  relative forms — hence `amt_ratio_median` outranking `amount_bdt` on gain.
- **Entity age is monotone in calendar time**, so raw `dev_age_days` drifts the
  same way a banned time index would. Each raw age is paired with a
  fraction-of-observable-history version.
- **"Suspicious clusters" are not connected components.** The lifetime
  customer–device graph is one giant component (93% of rows), and a 7-day
  recency window fragments it correctly (mean component size 3) but still yields
  ~1.0× lift. What *does* work is the learned link score: `gnn_cd_cos` reaches
  AP 0.219 (13× base). Component features are retained but carry no weight.
- **Missingness is MCAR** (NA-row fraud rates 0.0199 / 0.0193 / 0.0170 against a
  0.0176 base), so the NA level is kept explicit rather than imputed.
- Per-customer circular-time features (von Mises, Bahnsen et al. 2016)
  underperform a plain night flag here: fraud in this dataset is *globally*
  nocturnal rather than anomalous relative to each customer's own rhythm.

## Integrity note — `signup_inconsistency_d`

Comparing a row's implied signup day against the one the customer's earlier rows
established is, computed past-only, an ordinary data-quality feature. In real
fraud work "this account's stated age contradicts its own history" is a genuine
identity-tampering signal.

In *this* dataset it is near-deterministic: rows where it is non-zero are 85.8%
fraud (48.7× lift), covering ~7.5% of all fraud, while essentially no legitimate
row shows any inconsistency. That is a fingerprint of how the fraud rows were
synthesised, not behaviour a model is meant to learn — and the rules say
*"reverse-engineering the generative assumptions is not the intended path to a
good score"*, with reproducibility review able to flag work that leans on it.

It is **not** target leakage: it uses only raw non-target columns, past-only.
So this is a judgement call about the spirit of the rules, left explicit and
switchable rather than buried:

```python
# src/config.py
USE_SIGNUP_INCONSISTENCY = False   # default
```

It is worth roughly **+0.025 AP** (mean 0.790 with it, 0.765 without).

## References

- Bahnsen et al. (2016), *Feature engineering strategies for credit card fraud
  detection*, Expert Systems with Applications 51:134–142 — periodic (von Mises)
  time features and the RFM aggregation framing.
- Whitrow et al. (2009) — the transaction-aggregation window convention
  (1h/6h/24h/72h/168h) used in `velocity.py`.
- IEEE-CIS Fraud Detection (Kaggle, 2019) 1st-place solution — entity-level
  (UID) aggregation under temporal constraints.
