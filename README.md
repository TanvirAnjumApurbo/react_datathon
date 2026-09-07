# REACT 2026 Datathon — Leakage-Safe Preprocessing Pipeline

Feature engineering for the IEEE SEU SB REACT 2026 temporal fraud-detection
challenge. Metric: **PR-AUC (average precision)**.

The organisers ship only raw transaction fields and state that constructing
behavioural features *without looking into the future* is the competition. So
this repo is the preprocessing layer, not a model. The LightGBM run included
here exists only to confirm the split behaves and to rank the feature blocks.

## Leaderboard status

| submission | what changed | local `tail_late` | Δ local | public LB | Δ LB |
|---|---|---|---|---|---|
| 1 (2026-09-06) | 201 features, hill-climb blend | 0.516 | — | 0.52708 | — |
| 2 (2026-09-07) | 243 features, 7-day encoder delay, 5-config average | 0.5507 | +0.035 | **0.54151** | **+0.014** |
| 3 (2026-09-07) | + 8 recency/window variants (13 members) | 0.5530 | +0.0023 | 0.53901 | **−0.0025** |
| 4 (2026-09-07) | submission 2 + `signup_inconsistency_d` | ~+0.025 | +0.025 | **0.54874** | **+0.0072** |
| 5 (2026-09-07) | submission 4 + `cat_base` | — | +0.0011 | **0.55014** | **+0.0014** |
| 6 (2026-09-07) | submission 5 rebuilt deterministically, 3 seeds, pinned rounds | 0.5519 | ~0 | 0.54995 | **−0.00019** |
| 7 (2026-09-07) | same members, equal weight per *family* (CatBoost at ½) | 0.5526 | +0.0007 | **0.55150** | **+0.00155** |
| 8 (2026-09-07) | s7 with `signup_inconsistency_d` removed | 0.5511 | — | 0.54548 | −0.00602 |
| 9 (2026-09-07) | CatBoost weight 0.50 → 0.65 | 0.5448 | +0.0001 | **0.55187** | +0.00037 |
| 10 (2026-09-07) | CatBoost weight 0.65 → 0.80 | 0.5445 | −0.0003 | **0.55203** | +0.00016 |

Submission 1's gap was understood and was not a leak: the fraud amount signature
decays across the stream (fraud median 4,918 BDT in January, 944 in mid-July,
while the legitimate median holds at ~485), so the last two labelled weeks score
~0.53 locally -- matching the leaderboard -- while the six-week fold average
hides it. `primary_62d` was demoted for exactly this reason and `tail_late`
became the selection target.

**Four points give the calibration, and it has a threshold in it:**

* **Large local gains pass through at roughly 30–40%.** +0.035 local → +0.014
  board; +0.025 local → +0.0072 board.
* **Small local gains do not pass through at all — they are not even correctly
  signed.** Submission 3 gained +0.0023 on `tail_late` and *lost* 0.0025 on the
  board.

So `tail_late` is a sound proxy for changes worth roughly 0.008 AP or more, and
has **no resolving power at all** below that. This is stronger than a noise-floor
argument: a sub-0.005 local delta is not a small board gain, it is a coin flip.
Sixty-seven thousand rows and 1,068 positives cannot separate models that close,
and no amount of bootstrapping fixes it. Everything Stage 2 measured lives below
that threshold.

Submissions 3 and 5 make the point precisely. Both add members to a blend, both
have local deltas well under the limit, and they go opposite ways: eight
LightGBM variants that differ only in sample weights gained +0.0023 locally and
**lost** 0.0025 on the board; one CatBoost gained +0.0011 locally and **gained**
0.0014. What separates them is not the size of the local number but whether the
added members make *independent* errors. Add members for decorrelation, not for
count.

Full analysis, per-week numbers and the plan for the remaining submissions are in
**[SUBMISSIONS.md](SUBMISSIONS.md)**, which is the running decision log.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

python build_features.py --check     # build everything + prove past-only
python train_check.py                # walk-forward feature validation

python tune.py --set all --far       # Stage 2: which fitting procedure?
python blend.py                      # Stage 3: which weights, which post-process?
python submit.py --name my_run --seeds 2 --note "what this is spent to learn"
```

Before trusting any modelling number, run the diagnostics:

```bash
python baseline.py                   # raw-column reference AP -- the denominator
python adversarial_validation.py     # how different is test from train, and where
python ablate.py --set all           # does a feature change survive a paired test
```

The division of labour: `ablate.py` varies the **feature matrix**, `tune.py`
varies the **fitting procedure**, and both judge on `tail_late` with
`tail_recent` as confirmation and a paired bootstrap in place of a point
estimate. `submit.py` caches each member's seed-averaged test ranking, so a
second blend built from members already fitted costs seconds rather than another
refit — which is what makes it affordable to answer several questions in one
sitting.

Outputs land in `data/processed/`: `train_features.parquet`,
`test_features.parquet`, `feature_manifest.json`, plus `baseline_results.csv`,
`adversarial_features.csv`, `ablation_results.csv`, `tune_results.csv` and
`postprocess_results.csv`. Submissions are written to `submissions/<name>.csv`
with a `<name>.json` recording exactly what produced each one.

The label-free feature cache invalidates itself: `harness.load_base` keys it on
a hash of the feature-module sources, so editing a module rebuilds rather than
silently reusing the previous feature set.

## Results

243 features across seven blocks (`temporal` 21, `amount` 40, `velocity` 57,
`entity` 55, `behaviour` 28, `encoding` 13, `graph` 29). LightGBM,
expanding-window folds, target encoder rebuilt per fold with a 7-day feedback
delay:

| fold | horizon | val AP | vs base rate | train AP |
|---|---|---|---|---|
| **`tail_late`** | 17 d (46–62 ahead) | **0.5494** | 34.6× | 0.900 |
| `tail_recent` | 17 d (1–17 ahead) | 0.5611 | 35.3× | 0.881 |
| `primary_62d` | 62 d | 0.7340 | 43.7× | 0.903 |
| `wf_mar` | 30 d | 0.8177 | 42.8× | 0.935 |
| `wf_apr` | 30 d | 0.7903 | 45.6× | 0.961 |
| `wf_may` | 30 d | 0.7967 | 47.8× | 0.935 |

*(Measured before the stopping-window fix described below; re-running the same
single model afterwards gives `tail_late` 0.5509 / `tail_recent` 0.5593 /
`primary_62d` 0.7351, a change of 0.0002 on the selection fold. The walk-forward
rows have not been re-run.)*

**Do not average this column.** The six windows sit in two different fraud
regimes, and the mean (0.708) is the exact statistic that produced a 0.20
surprise on submission 1. `tail_late` is the number to read: it is the slice
that has tracked the leaderboard — 0.516 local against 0.527 public on the
previous feature set, 0.5507 against 0.54151 on this one. The rest measure
stability and horizon decay.

The spread is itself the finding: the same model scores 0.82 on March and 0.55
on the last two labelled weeks. That is not overfitting — train AP is roughly
flat at 0.88–0.96 across all six — it is the target moving.

Gain by block: amount 48.6%, velocity 27.4%, entity 13.1%, graph 3.4%,
temporal 2.9%, behaviour 2.4%, encoding 2.4%.

Gain share and marginal value are not the same thing, and the gap is
instructive. `encoding` carries 2.4% of gain, but removing its **feedback
delay** costs 0.008–0.028 AP on every window. `behaviour` also carries 2.4%,
and removing it entirely costs nothing measurable. Gain says how often a tree
split on something; the ablation says whether the model would miss it.

## Modelling

> **Superseded by Stage 2.** Everything in this section was measured on
> `primary_62d` with the 201-column feature set, and three of its conclusions
> have since been overturned on the tail folds: the config ranking, the
> exclusion of XGBoost and CatBoost, and the preference for a hill-climbed
> blend. It is kept because the submission-1 numbers came from here and because
> two of the reversals are instructive. Current numbers are under
> **[Stage 2](#stage-2--tuning-the-fitting-procedure-not-the-features)**.

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
  gets selected — which is exactly what happened to xgb and cat.

  > **Overturned.** On 243 features, `cat_base` is the best single model in the
  > project (0.7375 on this same fold, against the 0.7239 the exclusion was
  > written about) and equal-weighting all three families is a statistical tie
  > with LightGBM-only. The exclusion was correct about the models it measured
  > and wrong to be carried forward as a rule. And the hill-climb itself is now
  > rejected: it fits its weights on the rows the paired bootstrap resamples,
  > so the interval it reports cannot see its own selection bias.
- **Recency weighting does not pay.** Half-lives of 90 d and 45 d land within
  0.0006 AP of the unweighted model, so the drift in this data is not the kind
  that down-weighting old rows fixes. It does cut the round count roughly in
  half, which is a speed argument, not an accuracy one.

  > **Re-measured, and it holds.** The objection was that `primary_62d` averages
  > six weeks of one fraud-amount regime with two of the next and structurally
  > cannot see drift adaptation working. Re-run on `tail_recent`, whose training
  > data ends *inside* the new regime, across half-lives of 7/14/21/45/90 days
  > and hard windows of 45/90/120 days: **not one clears the noise floor.** The
  > right verdict for the wrong reason, and now for the right one. See
  > [Stage 2](#recency-weighting-does-not-pay-and-now-we-know-why).

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

## Stage 0 — the measuring instruments

Submission 1 scored 0.7252 locally and 0.52708 on the leaderboard. Nothing was
broken; the fold averaged a regime change away. Stage 0 built the instruments
that make that mistake hard to repeat, and ran the diagnostics that should have
come before any modelling.

### The tail window is now a fold

Two geometries, both scoring 2026-06-29 → 07-15 (67,196 rows, 1,068 positives):

| fold | cutoff | days ahead | what it answers |
|---|---|---|---|
| `tail_late` | 2026-05-14 | 46–62 | the LB-tracking slice; shares `primary_62d`'s cutoff, so it is a **free second read on the same fit** |
| `tail_recent` | 2026-06-28 | 1–17 | the only geometry whose training data ends inside the new regime, so the only one that can see recency adaptation working |

`harness.run_folds` groups folds by cutoff and fits each cutoff once, so
reporting the tail costs one extra masked AP call rather than a second model.

**The tail's measured 95% bootstrap band is ±0.027 AP** — wider than the ±0.02
previously assumed. A bare point estimate from it is not evidence. Comparisons
use `evaluate.paired_ap_delta`, a paired bootstrap over identical rows, which
cancels the shared sampling noise and puts the interval on the difference.

A third geometry, `tail_far` (cutoff 2026-04-15, 75–91 days ahead over the same
rows), is a **stress read and never a selection target**. The real test window
stops at 62 days, so `tail_far` asks a question nothing else can: of two configs
that `tail_late` cannot separate, which one is still standing when the drift has
had three months to work? That is the regime the September half of the test set
is in.

#### The early-stopping window was inside the window it was judging

The convention was "early-stop on the widest window at this cutoff, then report
the narrow ones as free reads on a model that was not tuned to them". It is a
good rule and it was applied to a case where it does not hold: `tail_late`
(06-29 → 07-15) is a **strict subset** of `primary_62d` (05-15 → 07-15), so
stopping on `primary_62d` chose the round count using 1,068 of the tail's own
positives — 28% of the stopping set. The tail was not the independent read it
was described as.

`validation.stopping_fold` now returns a registered window built to be disjoint
from every tail window (`es_late`, 05-15 → 06-28) and asserts the disjointness
rather than trusting the caller. `primary_62d` still overlaps its own stopping
window — no window at that cutoff can early-stop a 62-day fold without touching
it — which is accepted precisely because `primary_62d` is no longer selected on.

The contamination was *shared across candidates*, so the paired deltas in the
Stage 1 tables below are unaffected; it is the absolute levels that were
slightly optimistic. Numbers produced before the fix are marked where they
appear.

### The reference nobody had measured

`baseline.py` — a bare LightGBM on raw columns only:

| window | raw baseline | full pipeline | FE multiple |
|---|---|---|---|
| `tail_late` | 0.2259 | 0.5481 | 2.4× |
| `primary_62d` | 0.3167 | 0.7336 | 2.3× |
| `wf_mar` | 0.4238 | 0.8170 | 1.9× |

The engineering more than doubles the raw score, and it does so **on the tail as
much as on the primary fold** — so it is not buying old-regime performance and
paying for it in the test period. That was worth knowing rather than assuming.

The raw model also reproduces the weekly cliff, which is the cleanest evidence
that the collapse is in the data and not something the feature pipeline
introduced. With nothing but amount, hour, day of week, account age and the
five categoricals:

| week ending | 05-17 | 05-31 | 06-14 | 06-28 | **07-05** | **07-12** |
|---|---|---|---|---|---|---|
| raw baseline AP | 0.4325 | 0.3731 | 0.3250 | 0.3541 | **0.2402** | **0.1868** |
| lift over that week's base | 33.9× | 21.9× | 19.6× | 20.2× | 15.2× | **12.0×** |

Same shape as the full pipeline's, on a model that has no history features at
all to get stale.

**`scale_pos_weight` makes it worse.** Same features, same folds: unweighted
0.3167 vs `spw=55.8` 0.3071 on `primary_62d`, 0.2259 vs 0.2145 on `tail_late`,
and the unweighted model wins **6 of 6 windows**. Average precision reads only
the ordering, so upweighting the positive class adds no information and only
distorts the leaf values. It is well supported as an alternative to SMOTE; that
is not the same as being better than nothing.

### The drift is concept drift, not covariate shift

`adversarial_validation.py` trains a classifier to tell train rows from test
rows. On the raw columns it reaches AUC 0.612 — and **0.514, chance, once
`account_age_days` is removed**, which separates only because it counts one per
day for every returning customer. `amount_bdt` has PSI 0.0002 and KS 0.0030
between train and test: the amount distribution did not move.

What moved is `P(fraud | amount)`. That distinction settles a question rather
than raising one:

- **Density-ratio / inverse-propensity weighting cannot help.** It corrects a
  shift in `p(x)` while assuming `p(y|x)` is fixed — exactly backwards here.
  The weights it produces are nearly uniform anyway (p99 = 1.83, effective
  sample size 93.9%). Two independent LightGBM studies also found this exact
  correction losing to an untouched baseline (Pan et al., AdKDD 2020: IPW lost
  on all six drifted datasets by 1.7–4.5 AUC points; Qian et al.: 0.7202 vs a
  0.7237 baseline). The weights are still written to
  `adversarial_weights.parquet` so the claim stays checkable.
- **The PSI rule of thumb is unusable at this scale.** The 0.10/0.25 bands are
  Lewis (1994), calibrated for samples in the hundreds. PSI is asymptotically
  `(1/n + 1/m)·χ²_{B−1}`, so at 67k-vs-67k the 5% critical value is ~0.0005 —
  500× below the rule of thumb. PSI is reported as an effect size for ranking,
  with `psi_critical` giving the honest test.

### 44 features are `transaction_id` in disguise

On the **engineered** matrix the adversarial classifier reaches **AUC 1.0000**.
44 features are monotone in stream position *and* separable on their own; the
worst have test values almost entirely outside their train range:

| feature | adversarial AUC | train mean | test mean |
|---|---|---|---|
| `merch_age_days` | 0.997 | 98.7 | 226.2 |
| `dev_age_days` | 0.995 | 94.8 | 221.4 |
| `g_comp_size` | 0.981 | 38,547 | 57,356 |
| `gnn_cm_cos` | 0.933 | 0.571 | 0.958 |

Every test row sits past the largest split point the trees ever saw. The
iterative procedure (`--mode iterate`) had to strip **90 features** before the
AUC came near 0.75, and plateaued at 0.79 — the separability is pervasive.

`BANNED_FEATURES` was built to stop exactly this and does not catch any of it,
because none of them is literally a clock. Note also that `*_age_frac`, the
existing drift-stable counterpart, is not automatically safe: `dev_age_frac`
still separates at AUC 0.678 because it saturates toward 1 as the stream grows.

## Stage 1 — what the feature work actually bought

Every candidate below is one LightGBM fitted at the `primary_62d` cutoff and
read on both windows, with a paired bootstrap against the reference on the tail.
`ablate.py` produces this table; the verdict is deliberately conservative, since
a change has to clear the ~0.003 run-to-run floor *and* win the paired test.

| candidate | tail_late | primary_62d | tail Δ | 95% interval | verdict |
|---|---|---|---|---|---|
| reference (240 feat) | 0.5495 | 0.7344 | — | — | reference |
| **no encoder delay** | 0.5424 | 0.7182 | **−0.0071** | [−0.0105, −0.0036] | **WORSE** |
| drop all 44 time-counters | 0.5358 | 0.7161 | **−0.0137** | [−0.0203, −0.0077] | **WORSE** |
| drop `amount` block | 0.5426 | 0.7204 | −0.0069 | [−0.0137, −0.0010] | WORSE |
| drop `velocity` block | 0.5433 | 0.7227 | −0.0061 | [−0.0099, −0.0023] | WORSE |
| drop `entity` block | 0.5465 | 0.7286 | −0.0029 | [−0.0064, +0.0006] | no result |
| drop `encoding` block | 0.5505 | 0.7327 | +0.0011 | [−0.0016, +0.0041] | no result |
| drop `graph` block | 0.5514 | 0.7354 | +0.0020 | [−0.0006, +0.0044] | no result |
| drop `temporal` block | 0.5491 | 0.7345 | −0.0004 | [−0.0029, +0.0022] | no result |
| drop 7 severe counters | 0.5479 | 0.7319 | −0.0016 | [−0.0040, +0.0011] | no result |
| **drop everything Stage 1 added** | 0.5483 | 0.7334 | **−0.0012** | [−0.0041, +0.0014] | **no result** |
| drop `behaviour:habit` | 0.5496 | 0.7348 | +0.0001 | [−0.0020, +0.0025] | no result |
| drop `behaviour:repetition` | 0.5510 | 0.7345 | +0.0015 | [−0.0006, +0.0037] | no result |
| drop `behaviour:travel` | 0.5506 | 0.7339 | +0.0011 | [−0.0012, +0.0037] | no result |
| drop `behaviour:youth` | 0.5488 | 0.7336 | −0.0007 | [−0.0032, +0.0017] | no result |

Three things follow, and only one of them is a win.

### The feedback delay is the result

A fraud label does not exist when the transaction happens; it exists once an
investigation confirms it. Without a delay, a *training* row reads an encoder
that is perfectly up to date while a *test* row reads one frozen at the cutoff
and up to 62 days stale — so the model learns to trust the encoder more than it
will deserve at scoring time. Delaying the training rows' view closes the gap.

Confirmed on **every window**, not just the one it was measured on:

| delay | primary_62d | tail_late | tail_recent | wf_mar | wf_apr | wf_may |
|---|---|---|---|---|---|---|
| **7 d** (default) | 0.7336 | 0.5481 | 0.5622 | 0.8170 | 0.7894 | 0.7969 |
| none | −0.0154 | −0.0080 | −0.0065 | −0.0182 | −0.0277 | −0.0190 |
| 3 d | −0.0006 | −0.0003 | −0.0002 | −0.0012 | +0.0010 | −0.0013 |
| 14 d | −0.0013 | +0.0006 | +0.0002 | −0.0005 | +0.0030 | −0.0007 |
| 30 d | −0.0002 | +0.0004 | −0.0006 | +0.0004 | −0.0001 | −0.0010 |

Removing the delay costs **6 of 6 windows**, by 0.008 to 0.028 AP. Every delay
from 3 to 30 days is equivalent to within noise. So the effect is *having* a
delay, not tuning one — the damage comes from the encoder being perfectly fresh
for training rows, and three days of staleness is enough to remove it.
`config.TE_FEEDBACK_DELAY_D = 7.0`, following the ULB handbook.

### The new features are not

`drop_all_stage1` — removing all 42 columns this pass added — moves the tail by
−0.0012, inside the noise floor. The blocks measure the same way individually.

This is not for want of standalone signal. `amt_ratio_mean_30d`, the 30-day RFM
window the repo did not have, reaches AP 0.183 on the tail on its own (11.5×
base rate), second only to raw amount. It is simply **redundant**: the existing
`amt_ratio_median`, `amt_ratio_mean` and `amt_rank_30d` already carry that
information, and a tree does not care that a fourth column agrees with them.

Some of the additions have no signal at all, which is worth recording:

- **Round-number amounts do nothing here.** `amt_round_500`, `amt_cents` and
  `amt_has_cents` all sit at 1.00× base-rate lift. The mobile-money literature
  expects round figures to matter; in this data they do not.
- **Exact duplicates essentially do not occur.** Amounts are continuous to the
  paisa, so `(customer, merchant, amount)` never repeats inside an hour — that
  column was identically zero across all 994,590 rows and is dropped at build
  time. The 24h and 7d versions fire on ~0.02% of rows at 1.00× lift.
- **Entropy is weak; surprise is better.** Normalised entropy over a customer's
  category history reaches 1.07–1.36× lift, while the self-information of the
  observed category (`cust_dtype_surprise`) reaches 2.41×. If the habit family
  is revisited, the surprise form is the one to keep.
- **A binary flag is not an interaction.** `is_young_account` alone is 1.00×;
  `young_acct_risk`, which combines it with an unusual amount and an unfamiliar
  device, is 4.96×. Single-feature AP systematically understates anything that
  only matters in combination, which is why the model-level ablation is the
  arbiter and not the standalone scan.

They are kept — leakage-clean, cheap, and useful as ensemble diversity — but
they are recorded as *no result*, not as an improvement.

### Dropping the drift-flagged counters is actively harmful

The adversarial diagnostic says 44 features separate train from test almost
perfectly. The obvious move is to drop them. Measured, that costs **−0.0137 on
the tail** and −0.0183 on `primary_62d`, both well outside the interval.

So a feature can be a near-perfect train/test discriminator and still be worth
carrying. The trees appear to use these counters for within-period ordering,
which survives even when every test row lands in the rightmost bin. This is the
over-dropping failure Pan et al. record (a 12-point AUC collapse on one of their
datasets), reproduced here.

### A worked example of selecting on the evaluation window

Two candidates were sub-threshold positives: dropping the `graph` block
(+0.0020) and dropping the new amount columns (+0.0022). Combining them, and
then keeping only the blocks with significant verdicts, looked like a real gain:

| candidate | tail_late | primary_62d | verdict on the tail |
|---|---|---|---|
| reference (243 feat) | 0.5481 | 0.7336 | reference |
| drop graph + new amount (200) | 0.5513 | 0.7349 | **BETTER** (+0.0033) |
| significant blocks only (172) | 0.5525 | 0.7362 | **BETTER** (+0.0045) |

Both clear the paired test. Then check the windows they were *not* selected on:

| candidate | tail_recent | wf_mar | wf_apr | wf_may |
|---|---|---|---|---|
| drop graph + new amount | **−0.0022** | +0.0003 | +0.0006 | −0.0027 |
| significant blocks only | **−0.0035** | −0.0004 | +0.0001 | −0.0009 |

Flat to negative everywhere else. `tail_recent` is the sharpest disconfirmation
available: it scores the *identical rows*, differing only in that the model was
trained six weeks later — and it disagrees in sign.

The gain was selection bias. The feature set was chosen by reading `tail_late`,
so it fits `tail_late`. **Neither trim is adopted.** The lesson generalises to
anything else picked off the tail: the tail is small, and choosing against it
repeatedly will overfit it exactly the way chasing a public leaderboard does.

## Stage 2 — tuning the fitting procedure, not the features

Run with `tune.py`, which differs from `ablate.py` in what it varies: the model
and how it is fitted, rather than which columns it sees. Selection is on
`tail_late`, confirmation on `tail_recent` (the same rows from a later cutoff),
and a candidate that helps one while hurting the other is recorded as **SPLIT**
rather than adopted.

### Hyperparameters do not matter here

Five LightGBM configurations spanning the ranges the literature recommends —
`learning_rate` 0.03–0.05, `num_leaves` 31–255, `min_data_in_leaf` 50–300,
`feature_fraction` 0.5–0.8, `lambda_l1/l2` 0–10, plus extremely randomized
splits:

| config | tail_late | tail_recent | primary_62d | verdict |
|---|---|---|---|---|
| `lgb_base` (64 leaves, lr 0.05) | **0.5509** | 0.5593 | 0.7351 | reference |
| `lgb_shallow` (31 leaves) | 0.5503 | 0.5595 | 0.7350 | no result |
| `lgb_deep` (255 leaves, lr 0.03) | 0.5498 | **0.5607** | 0.7326 | no result |
| `lgb_reg` (heavy L1+L2, min_data 300) | 0.5472 | **0.5615** | 0.7327 | SPLIT |
| `lgb_extra` (extremely randomized) | 0.5449 | 0.5595 | 0.7307 | SPLIT |

Total spread 0.0060 AP on a window whose own 95% band is ±0.027. Nothing here
is a result, and the two configs that move furthest move *in opposite
directions on the two folds* — which is the signature of noise, not of a
tuning gradient.

There is one suggestive pattern, held loosely because it does not clear the
floor: regularization ranks worst at 46–62 days ahead and best at 1–17. If real,
it says a heavily regularized model has less to lose from being stale, which is
the wrong trade when the model *will* be stale. The feature work bought 2.3–2.4×
over raw columns; the hyperparameters buy nothing.

### Recency weighting does not pay, and now we know why

This was the repo's largest open question. The earlier verdict was measured on
`primary_62d`, whose 62-day window averages two fraud regimes and structurally
cannot see drift adaptation working, so it was marked superseded. `tail_recent`
(cutoff 2026-06-28, training data ending *inside* the new regime) can see it.
Both knobs, scored on all three geometries:

| candidate | tail_late | tail_recent | tail_far | verdict |
|---|---|---|---|---|
| `lgb_base` (no adaptation) | 0.5509 | 0.5593 | 0.5304 | reference |
| `hl_21d` | 0.5535 | 0.5586 | 0.5290 | no result |
| `hl_14d` | 0.5531 | 0.5592 | 0.5313 | no result |
| `hl_45d` | 0.5511 | 0.5608 | 0.5299 | no result |
| `hl_7d` | 0.5501 | 0.5545 | 0.5236 | unclear |
| `hl_90d` | 0.5500 | **0.5626** | 0.5315 | SPLIT |
| `win_45d` (24% of the data) | 0.5516 | 0.5599 | 0.5312 | no result |
| `win_90d` | 0.5493 | 0.5624 | 0.5311 | SPLIT |
| `win_120d` | 0.5498 | 0.5608 | 0.5305 | no result |

Half-lives from 7 to 90 days and hard windows from 45 to 120 days: **not one
clears the noise floor**, on the fold built to detect exactly this. The verdict
is no longer superseded — it is confirmed, on the right evidence.

The explanation is the interesting part. **`win_45d` trains on 175,571 rows —
24% of the labelled data — and scores the same as training on all 731,942.**
Three quarters of the training set contributes nothing measurable. That is why
recency weighting cannot help: it exists to down-weight stale rows, and the
model was already ignoring them. The signal here is short-memory, and a
weighting scheme has nothing left to correct.

### CatBoost was written off on stale evidence

"Do not equal-weight ensemble across model families" was recorded when XGBoost
and CatBoost both scored below every LightGBM config. Re-measured on 243
features and read on the tail, that is no longer true:

| member | tail_late | tail_recent | primary_62d |
|---|---|---|---|
| `cat_base` | **0.5530** | 0.5608 | **0.7375** |
| `lgb_base` | 0.5509 | 0.5593 | 0.7351 |
| `xgb_base` | 0.5482 | **0.5620** | 0.7337 |
| `xgb_shallow` | 0.5467 | 0.5606 | 0.7322 |

CatBoost is now the **best single model in the project** on both `tail_late` and
`primary_62d` (0.7375 against the 0.7239 the old note was written about), and
XGBoost is the best of anything at the short horizon. None of the gaps clears
the noise floor, so this is not "switch to CatBoost" — it is "the reason for
excluding two thirds of the model families no longer holds, and an ensemble may
now have three families to draw on instead of one."

### The ensemble ladder, and the rule that had to be rewritten mid-run

Blend options divide into two classes that must not be judged on the same terms:

* **Unweighted** — one model, or an equal average over an a-priori group.
  Nothing about them was chosen by looking at the tail, so the measured
  objective is an honest estimate and the best one simply wins.
* **Fitted** — the Caruana hill-climb, whose weights are optimised on the very
  1,068 positives the paired bootstrap then resamples. The bootstrap cannot see
  that bias.

The first version of this ladder ignored the distinction: it started from the
simplest option and adopted anything that beat it by more than the noise floor.
Run over 16 members it **adopted the hill-climb** — which had put weight on
`hl_7d`, individually the *worst* member of the pool. A greedy search reaching
for a bad-but-decorrelated member to squeeze one specific window is the
signature of overfitting, and `significant_blocks_only` is the receipt: it
cleared a paired test at +0.0045 on `tail_late` and went negative on
`tail_recent`.

Rewritten: pick the best unweighted option outright, and require a fitted blend
to beat *that* by twice the floor. The verdict inverts.

| option | members | tail_late | tail_recent | objective |
|---|---|---|---|---|
| `equal_lgb_all` | 13 LightGBM | 0.5530 | 0.5630 | **0.5580** |
| `equal_all` | 16, all families | 0.5529 | 0.5630 | 0.5579 |
| `best_single` (`cat_base`) | 1 | 0.5530 | 0.5608 | 0.5569 |
| `equal_core` (submission 2) | 5 | 0.5507 | 0.5615 | 0.5561 |
| `hill_climb` | 6, fitted | 0.5543 | 0.5633 | 0.5588 |

The hill-climb still has the highest objective. It beats `equal_core` by +0.0036
and the *best unweighted option* by +0.0008 — and the second number is the one
that matters. Rejected. `equal_lgb_all` is adopted: thirteen equally weighted
members, not one weight learned from the tail. It ties `equal_all` at 0.0001,
which is not a distinction; the rule picks by objective rather than letting
taste in through a gap that small.

### Entity post-processing: the version that works is the version that cheats

The IEEE-CIS winners replaced every prediction for a client with that client's
mean. Ported here as `p' = (1-a)·p + a·(entity statistic)` over `customer`,
`merchant` and `device`, with four statistics and six values of `a`:

| statistic | what it assumes | tail_late delta |
|---|---|---|
| `mean` | the entity is broadly risky | **−0.09 to −0.28** |
| `max` (over all scored rows) | one bad transaction condemns the rest | +0.0040 at a=0.5, +0.0044 at a=0.7 |
| `cummax` (past-only) | same, using only the entity's earlier rows | **+0.0004** |

Two findings, and the second is the one that matters.

**Mean is catastrophic**, by −0.09 to −0.28 AP. It is the natural port of the
IEEE-CIS move and it fails for a structural reason: there, the label was a
property of the client, so pooling within a client added information. Here fraud
is per-transaction and a customer's fraudulent rows are a small minority of
their own history, so averaging deletes exactly the within-entity ordering that
average precision is computed from.

**Max looks like a real gain — until it is made causal.** `device`/`max` at
a=0.5 clears the floor with a paired interval of [+0.0019, +0.0063]. It was
adopted, then re-run as `cummax`, which restricts each row to its device's
*earlier* rows. The gain went to +0.0004: **nothing**. So the entire effect came
from a July test row being adjusted by a September one, which is the two-sided
aggregation the rules exclude — they permit non-target aggregation over test
rows only "in a strictly-past-only way". There is no fraud-ring signal being
recovered here; there is only the future. **No post-processing is adopted**, and
`blend.py` now refuses to adopt any non-causal variant regardless of its score.

## Stage 4 — the far horizon, and four members that did not earn a place

`tune.py --far` now covers every candidate, so all 20 members carry three reads
over the *same* 67k rows: `tail_recent` (1–17 days ahead), `tail_late` (46–62)
and `tail_far` (75–91). The real test window spans 1–62 days past its cutoff and
then keeps drifting for two more months of calendar time, so it sits between the
last two.

### The horizon reorders the members

| member | tail_recent | tail_late | tail_far | decay |
|---|---|---|---|---|
| `cat_base` | 0.5608 | **0.5516** | **0.5328** | 0.0188 |
| `lgb_extra` | 0.5590 | 0.5475 | **0.5347** | **0.0128** |
| `lgb_shallow` | 0.5601 | 0.5524 | 0.5320 | 0.0204 |
| `lgb_base` | 0.5600 | 0.5491 | 0.5320 | 0.0169 |
| `xgb_base` | **0.5625** | 0.5481 | 0.5305 | 0.0176 |
| `lgb_deep` | 0.5616 | 0.5509 | 0.5299 | 0.0210 |

`lgb_extra` is 7th of 8 at `tail_late` and first at `tail_far`; `lgb_deep` is
second at `tail_recent` and last at `tail_far`. Reading one horizon is not
merely noisier than reading three — it answers a different question than the
private half of the test set asks.

**Blend decay rises with member count**, which gives submission 3's leaderboard
loss a mechanism: `equal_core` (5 members) decays 0.0169, `equal_lgb_all` (13)
decays 0.0197 while scoring *higher* at `tail_late`, and `equal_all` (20) has
the worst far read of any composition. Extra members buy near-horizon AP and pay
for it in the regime that decides the private 40%.

### Four members proposed with mechanisms; three refuted

Each was specified before measurement and judged on a four-part gate: mechanism,
decorrelation (rank ρ ≤ 0.70 against the blend), competence (within 0.010 of the
core at `tail_late`), and horizon (decay no worse than the core's).

| member | ρ | tail_late | tail_far | decay | verdict |
|---|---|---|---|---|---|
| `lgb_durable` (no `amount`) | 0.777 | 0.5397 | 0.5197 | 0.0200 | refuted |
| `cat_durable` | 0.632 | 0.5385 | 0.5143 | 0.0242 | refuted |
| `lgb_linear` (`linear_tree`) | 0.572 | 0.5484 | **0.5016** | 0.0468 | refuted |
| `lgb_sub30` (`feature_fraction=0.3`) | 0.829 | 0.5483 | 0.5331 | 0.0152 | clone |

**Standalone feature retention does not predict model behaviour.**
`lgb_durable` was built because the amount family keeps only 0.61–0.64 of its
power across the regime split while `cust_dt` keeps 0.89 and `gnn_cd_cos` 0.99 —
so a model denied that block should decay more slowly. It decays *faster*
(0.0200 vs 0.0169) and is 0.0094 worse at `tail_late`. `cat_durable` inherits
the same weakness through a different algorithm, so it is the feature view and
not the learner.

**`lgb_linear` is why `tail_far` exists.** It ties the core on the selection
target, clears the decorrelation gate at ρ 0.572, and has a documented
mechanism — it would have passed every check this project had. At 75–91 days it
collapses to 0.5016, a decay nearly 3× anything else measured. Linear leaves
extrapolate as linear functions and diverge once the unbounded counters leave
their training range. (It also stopped at 60 rounds, so under-training competes
as an explanation; it fails either way.)

**Decorrelation and competence trade off.** The two most independent members in
the project (`lgb_linear` 0.572, `cat_durable` 0.632) are exactly the two that
fall apart at the far horizon, and the one that matches the core at every
horizon is a clone at 0.829. `cat_base` — ρ 0.624 *and* competitive everywhere —
is the only exception, which is why it is the one member that ever paid.
`xgb_base` sits at ρ 0.824, inside the LightGBM clone band: **a different
library is not automatically a different model.**

### `mean(tail_late, tail_far)` is a better estimator and still not a discriminator

| composition | tail_late | mean(late,far) | board | err(late) | err(mean) |
|---|---|---|---|---|---|
| `equal_core` (= s2) | 0.5511 | 0.5427 | 0.5415 | +0.0096 | **+0.0012** |
| `equal_lgb_all` (= s3) | 0.5532 | 0.5433 | 0.5390 | +0.0141 | **+0.0043** |

Eight times closer on the *level*, and still wrong on the *difference*: the
board puts s2 above s3 by 0.0025 and the mean puts s3 above s2 by 0.0007. So the
0.008 resolution limit is not an artefact of a poorly-centred statistic. 1,068
positives cannot separate models 0.0025 apart, whatever functional is computed
from them. `horizon_proxy.py` runs this test and prints the verdict.

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
  validation.py          forward-block folds incl. the two tail folds
  evaluate.py            AP + lift, weekly/horizon breakdowns, bootstrap bands
  harness.py             fold running (one fit per cutoff) + the feature cache
  leakage_checks.py      the guards described below
  features/
    _windows.py          the past-only window primitive everything routes through
    temporal.py          hour/day, cyclical, night, per-customer circular rhythm
    amount.py            amount vs own history; RFM windows; trailing ranks
    velocity.py          customer/device/merchant recency + rolling windows
    entity.py            pair novelty, device sharing, diversity, movement
    behaviour.py         habit/surprise, repetition, travel rate, account youth
    encoding.py          past-only target encoding, frozen cutoff + feedback delay
    graph.py             snapshot bipartite graph: degrees, components, spectral
    gnn.py               self-supervised temporal GNN embeddings (Tier C)
build_features.py        orchestrator
train_check.py           walk-forward feature validation harness
baseline.py              raw-column reference AP -- the denominator
adversarial_validation.py  train-vs-test drift: AUC, PSI, KS, drop shortlist
ablate.py                does a change help on the tail? paired-bootstrap verdicts
tune.py                  does a change to the *fitting procedure* help
blend.py                 weights + post-processing, with the decorrelation table
horizon_proxy.py         which local statistic predicts the board? fits nothing
submit.py                refit on all labelled data, write submissions/<name>.{csv,json}
make_notebook.py         emit the rulebook-8.2 reproducibility notebook
notebooks/               generated per-submission reproduction notebooks
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

### The dominant run-to-run variance source is the feature build, not the model

This was measured wrong at first, and the correction matters. `lgb_deep`
early-stopped at **230 rounds** on one day and **758** on the next — same code,
same data, same params, same disjoint stopping window — for a `tail_late` change
of 0.0011. The natural reading is LightGBM's thread non-determinism. It is not:
`lgb_base` reproduced to four decimals across three separate processes in one
session. What changed between the two days was that the feature cache had been
rebuilt.

| `gnn_*` column | bit-identical across builds | max abs diff | correlation |
|---|---|---|---|
| `gnn_cd_score` | no | 3.2e-04 | 1.000000 |
| `gnn_cd_cos` | no | 1.5e-05 | 1.000000 |
| `gnn_cm_cos` | no | 1.0e-05 | 1.000000 |
| `gnn_cust_emb_drift` | no | 3.0e-06 | 1.000000 |

`gnn.py` is properly seeded (`torch.manual_seed`, seeded generators, a seeded
state initialiser); this is float non-determinism in multi-threaded CPU
reductions. **A 3e-4 perturbation of four columns out of 244 moves an early stop
by 3×**, which says the average-precision stopping curve here is a plateau
rather than a peak.

Three consequences:

- **Round counts must be pinned, not re-derived.** `submit.py --rounds-book`
  takes the book a submission was built from; naming the same members does not
  identify the same model.
- **Reproducing a submission needs the matrix, not just the code.** A rerun
  ranks the test set essentially identically (feature correlation 1.000000) and
  scores within the noise floor, but the CSV is not byte-identical unless the
  built matrix is shipped alongside the notebook.
- **A paired-bootstrap verdict is not stable across a rebuild.** `lgb_shallow`
  measured −0.0007 against `lgb_base` on one build and **+0.0032, verdict
  `BETTER`, interval excluding zero** on the next. The verdict machinery
  promoted a candidate on early-stopping churn alone — the sharpest evidence yet
  that sub-0.008 deltas are not results.

### Run-to-run variation within a single feature build

LightGBM is seeded (`seed`, `bagging_seed`, `feature_fraction_seed`) but runs
with `num_threads=0` and without `deterministic=True`, so histogram accumulation
order can vary between threads and tiny float differences compound over 1,600
boosting rounds. The measurement below was taken across two runs that also
differed in their feature matrix, so it conflates the two sources and should be
read as an upper bound on the model's own contribution; within one build,
LightGBM has reproduced exactly here.

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
- **A stale encoder beats a fresh one.** Giving the target encoder a 7-day
  feedback delay — so a training row reads labels as they stood a week earlier,
  the way a deployed system would — is worth 0.008 to 0.028 AP on all six
  windows. Without it, training rows see a perfectly current encoder while test
  rows see one frozen at the cutoff, and the model calibrates its trust to a
  freshness it will not get. Any delay from 3 to 30 days works equally well.
- **`scale_pos_weight` costs AP.** Unweighted beats `spw = 55.8` on 6 of 6
  windows. Average precision reads only the ordering, so upweighting the
  positive class adds no information — it only distorts the leaf values.
- **A perfect train/test discriminator can still be worth keeping.** 44
  features separate the two periods at up to AUC 0.997 because they are
  unbounded counters; dropping them costs 0.014 AP on the tail. The adversarial
  diagnostic is good at *finding* drift-exposed features and bad at deciding
  their fate.
- **Round-number amounts and exact duplicates carry nothing here.**
  `amt_round_500`, `amt_cents` and the duplicate counters all sit at ~1.00×
  base-rate lift, and `(customer, merchant, amount)` never repeats inside an
  hour across 994,590 rows, because amounts are continuous to the paisa. The
  mobile-money literature expects both to matter; this generator does not
  produce them.
- **Surprise beats entropy.** The self-information of the observed category
  under a customer's own history reaches 2.4× lift; the entropy of that history
  reaches 1.4×. "How unusual is *this* choice" is a sharper question than "how
  varied is this customer".

## Integrity note — `signup_inconsistency_d`

Comparing a row's implied signup day against the one the customer's earlier rows
established is, computed past-only, an ordinary data-quality feature. In real
fraud work "this account's stated age contradicts its own history" is a genuine
identity-tampering signal.

In *this* dataset it is near-deterministic: rows where it is non-zero are 85.8%
fraud (48.7× lift), covering 6.8% of all fraud, while essentially no legitimate
row shows any inconsistency. That is a fingerprint of how the fraud rows were
synthesised, not behaviour a model is meant to learn — and the rules say
*"reverse-engineering the generative assumptions is not the intended path to a
good score"*, with reproducibility review able to flag work that leans on it.

It is **not** target leakage: it uses only raw non-target columns, past-only.
So this is a judgement call about the spirit of the rules, left explicit and
switchable rather than buried:

```python
# src/config.py -- default off, resolved from the environment at import time
USE_SIGNUP_INCONSISTENCY = os.environ.get("REACT_SIGNUP", "").strip().lower() in {
    "1", "true", "yes"
}
```

```bash
REACT_SIGNUP=1 .venv/Scripts/python.exe -u submit.py --name with_signup ...
```

The environment override exists because the earlier arrangement — a literal in
tracked source, flipped to `True` for a build and back afterwards — left the repo
on the wrong value twice when a build process was killed before it could restore
it. A build flag that requires mutating tracked source is a flag that will
eventually be left in the wrong state. The resolved value is folded into
`harness.feature_source_hash`, so the feature cache invalidates correctly either
way, and the generated reproduction notebook sets the variable from the
submission's own manifest so it cannot be run in a configuration that differs
from the one it claims to reproduce.

It is worth roughly **+0.025 AP** locally, and **+0.0072 measured on the
leaderboard** (submission 4 against submission 2, one variable changed).

## References

- Bahnsen et al. (2016), *Feature engineering strategies for credit card fraud
  detection*, Expert Systems with Applications 51:134–142 — periodic (von Mises)
  time features and the RFM aggregation framing.
- Whitrow et al. (2009) — the transaction-aggregation window convention
  (1h/6h/24h/72h/168h) used in `velocity.py`.
- IEEE-CIS Fraud Detection (Kaggle, 2019) 1st-place solution — entity-level
  (UID) aggregation under temporal constraints.
