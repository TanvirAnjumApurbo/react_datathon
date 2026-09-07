# Submission log — REACT 2026

Budget: **10 submissions total**, max 5/day. Leaderboard is **60% public / 40%
private**; up to **2 submissions** may be selected for private scoring (default:
best public). The rules say explicitly: *"Do not chase the public leaderboard…
Trust a robust, time-aware local validation setup over repeated public
submissions."*

So each entry below records not just the score but **what question the
submission was spent to answer**. A submission that answers nothing is wasted
even if it scores well.

| # | date | file | local `tail_late` | public LB | Δ vs local | question it answered |
|---|---|---|---|---|---|---|
| 1 | 2026-09-06 | `submission.csv` `ab115128…` | 0.516 (0.7252 primary) | **0.52708** | +0.011 | Does local validation predict the leaderboard? **Not `primary_62d` — but the tail does.** |
| 2 | 2026-09-07 | `s2_pipeline_equal_lgb` | 0.5507 | **0.54151** | −0.009 | Does the Stage 0/1 rebuild move the board? **Yes, +0.014 — and local gains arrive at ~40% size.** |
| 3 | 2026-09-07 | `s3_equal_lgb_all` | 0.5530 | **0.53901** | −0.014 | Does averaging 13 differently-memoried models help? **No — it cost 0.0025, while local said +0.0023.** |
| 4 | 2026-09-07 | `s4_signup_integrity` | ~+0.025 est | **0.54874** | — | What is `signup_inconsistency_d` worth? **+0.0072. Real, priced, and still not adopted.** |
| 5 | 2026-09-07 | `s5_signup_plus_cat` | — | **0.55014** | — | Submission 4 + `cat_base`: does one decorrelated member beat eight clones? **Yes, +0.0014 — and local called this one correctly.** |
| 6 | 2026-09-07 | `s6_det_s5comp` | 0.5519 | **0.54995** | −0.002 | Is the leading submission reproducible, and what does a same-composition rebuild cost? **0.00019** — the board's floor is 10x tighter than assumed. |
| 7 | 2026-09-07 | `s7_family_equal` | 0.5526 | **0.55150** | −0.001 | Weight per *family*, not per member. **+0.00155 over s6 from the weight vector alone — the best score in the project.** |
| 8 | 2026-09-07 | `s8_clean_line` | 0.5511 | **0.54548** | — | What is `signup_inconsistency_d` worth at the **final** composition? **+0.00602**, against +0.0072 on the older blend. |
| 9 | 2026-09-07 | `s9_cat65` | 0.5448 | **0.55187** | — | Is the board's blend-weight optimum right of local's? **Yes: w_cat 0.50 → 0.65 gained +0.00037.** |
| 10 | 2026-09-07 | `s10_cat80` | 0.5445 | **0.55203** | +0.008 | Where does the weight curve turn? **It does not, by w=0.80 — best score of the competition.** |

---

## Submission 1 — baseline hill-climb blend

**Date:** 2026-09-06 · **Public LB: 0.52708** · local `primary_62d` AP 0.7252

### What was submitted

201 features across six blocks (amount 47%, velocity 27%, entity 12%, encoding
7%, graph 4%, temporal 3%), built past-only over the combined train+test stream.
A 15-model LightGBM blend — `lgb_deep` x3, `lgb_hl90` x3, `lgb_hl45` x3,
`lgb_base` x3, `lgb_shallow` x3 — with weights (0.459 / 0.189 / 0.189 / 0.108 /
0.054) chosen by greedy hill-climb on the primary fold. Rank-blended, then
quantile-mapped to a probability range (monotone, so score-neutral).

All leakage guards green: truncation invariance, no banned columns, no
near-perfect single feature (best honest one is `amt_ratio_median` at 0.44),
boundary continuity, end-of-stream coverage. `USE_SIGNUP_INCONSISTENCY = False`.

### Result

**Local said 0.7252. The leaderboard said 0.52708.** A gap of ~0.20 AP.

### What it told us — the drift is in the *fraud generator*, not the population

The gap is not noise, not a leak, and not a base-rate artifact. Three checks:

**1. It is not prevalence.** Reweighting train's fraud rate by the test amount
distribution gives 0.0173, and by the test hour distribution 0.0174, against a
train base rate of 0.0176. The test period's fraud rate is very likely still
~1.7%, so the LB score reflects a genuine ~30% loss of ranking power, not a
denominator change. (The submission's mean predicted probability of 0.0119 is
**not** evidence of anything — the calibration step is a pure quantile map onto
the validation distribution, so that mean is inherited from the fold by
construction.)

**2. The loss was already visible inside our own validation fold**, and we
averaged it away. Breaking the fold down by forecast horizon:

| days after fold cutoff | n | base | AP | lift |
|---|---|---|---|---|
| 0–7 | 26,430 | 0.0173 | 0.8239 | 47.5x |
| 7–15 | 30,413 | 0.0139 | 0.7471 | 53.8x |
| 15–31 | 61,503 | 0.0175 | 0.7806 | 44.6x |
| 31–46 | 58,396 | 0.0188 | 0.8087 | 43.1x |
| **46–62** | **63,323** | **0.0154** | **0.5161** | **33.5x** |
| all | 240,065 | 0.0168 | 0.7253 | 43.2x |

It is not gradual decay — it is flat for six weeks and then a cliff. Weekly:

| week ending | AP | week ending | AP |
|---|---|---|---|
| 2026-05-24 | 0.8105 | 2026-06-28 | 0.7903 |
| 2026-05-31 | 0.8074 | **2026-07-05** | **0.5986** |
| 2026-06-07 | 0.7634 | **2026-07-12** | **0.4729** |
| 2026-06-21 | 0.8142 | | |

**The last two labelled weeks average ≈ 0.53. The public LB is 0.52708.** The
leaderboard is not a surprise; it is the number that was sitting in the tail of
our own training data, hidden under a six-week average.

**3. The mechanism: fraud is converging toward legitimate behaviour in amount
space.** Legitimate median amount is *rock stable* at ~485 BDT for all 37 weeks.
Fraud amounts collapse:

| week | fraud median | fraud P90 | legit median | AP of raw `amount_bdt` |
|---|---|---|---|---|
| 2026-01-11 | 4,918 | 35,873 | 481 | 0.4547 |
| 2026-03-08 | 4,404 | 34,381 | 487 | 0.4250 |
| 2026-05-10 | 2,044 | 21,992 | 492 | 0.2773 |
| 2026-06-14 | 1,426 | 24,649 | 480 | 0.2437 |
| **2026-07-12** | **944** | **12,620** | 486 | **0.1485** |

Raw amount loses **3x** of its standalone power across the stream. Because the
stream-wide amount tail keeps falling right through the test period (weekly
P99.9 is 42k–56k in January, 27k in mid-July, and **18k–29k for every test
week**), the trend does not stop at the train boundary — the test period is
*entirely* inside the new regime, and drifting further.

**And the "drift-robust" relative amount forms do not escape it.** Measured
early (05-15 → 06-28) vs late (06-29 → 07-15) inside the fold:

| feature | early AP | late AP | retained |
|---|---|---|---|
| `amt_ratio_median` | 0.4332 | 0.2752 | 0.64 |
| `log_amt_ratio_mean` | 0.4206 | 0.2570 | 0.61 |
| `amt_ratio_last20` | 0.4150 | 0.2514 | 0.61 |
| `amount_bdt` | 0.2664 | 0.1713 | 0.64 |
| `dev_prior_ncust` | 0.0561 | 0.0344 | 0.61 |
| `cust_n_1h` | 0.1588 | 0.1178 | 0.74 |
| `cust_dt` (recency) | 0.0545 | 0.0485 | **0.89** |
| `gnn_cd_cos` | 0.0118 | 0.0117 | **0.99** |

Ratios-to-own-history were designed to absorb a shift in the *population*
amount distribution. This is a shift in the *fraud* distribution — fraud moved
to smaller amounts — so normalising by the customer's own history does not help.
**Velocity/recency and graph features are the resilient ones.**

### What changed as a result

1. `primary_62d` is now known to **overstate by ~0.20 AP**. It is still fine for
   ranking configurations against each other, but it is not an LB predictor.
2. A new validation target is needed: **the last ~2.5 weeks of train
   (2026-06-29 → 2026-07-15, ~67k rows, ~1,050 positives)**. That window scores
   0.516 and the LB scored 0.527 — it is the only slice that tracks reality.
   It is small, so treat its noise band as wide (~±0.02), but it is the right
   target.
3. The conclusion **"recency weighting does not pay"** in `README.md` is now
   suspect. It was measured against a six-week old-regime average, which is
   exactly the objective that cannot see this. It must be re-measured against
   the tail window before being trusted.

---

## Between submissions 1 and 2 — Stage 0 and Stage 1, no submissions spent

Three of the four hypotheses queued for submissions 2–4 were answerable
locally, and two came back negative. That is the point of building the
instruments first: **a question you can settle offline should never cost a
submission.**

### What was built

| tool | question it answers |
|---|---|
| `tail_late` / `tail_recent` folds in `config.py` | is the tail still a manual recomputation? no — it is a fold, and `tail_late` shares `primary_62d`'s cutoff so it is free |
| `src/evaluate.py` | AP with lift, weekly and horizon breakdowns, bootstrap bands, **paired** bootstrap for A/B |
| `src/harness.py` | one fit per cutoff, encoder rebuilt per fold, cache keyed on a hash of the feature-module sources |
| `baseline.py` | what does the raw data give for free? |
| `adversarial_validation.py` | how different is test from train, and where? |
| `ablate.py` | does a change survive a paired test on the tail? |

### What was measured

**The tail's noise band is ±0.027 AP, not ±0.02.** Wider than assumed. Single
point estimates off the tail are not evidence; the paired bootstrap is.

**Feature engineering is worth 2.3–2.4× over raw columns, and it holds on the
tail.** Raw-column LightGBM: 0.2259 tail / 0.3167 primary, against 0.5481 /
0.7336 for the pipeline. The engineering is not buying old-regime performance
at the expense of the test period.

**`scale_pos_weight` costs AP.** Unweighted beats `spw = 55.8` on 6 of 6
windows (0.3167 vs 0.3071 primary, 0.2259 vs 0.2145 tail). AP reads only the
ordering, so upweighting positives adds nothing and distorts the leaf values.

**The drift is concept drift, not covariate shift.** Adversarial AUC on the raw
columns is 0.612, and **0.514 — chance — with `account_age_days` removed**;
`amount_bdt` has PSI 0.0002 between train and test. The amount *distribution*
did not move; `P(fraud | amount)` did. Consequence: **density-ratio importance
weighting cannot work here** and is off the table, which removes one of the
four planned lines of attack. (Two independent LightGBM studies also found it
losing to an untouched baseline.)

### Hypotheses closed without a submission

| planned # | hypothesis | verdict |
|---|---|---|
| 3 | "the amount family is actively harmful late" | **False.** Dropping the `amount` block costs −0.0069 on the tail and −0.0140 on primary. It is the strongest block, drift and all. |
| — | "drop the features that separate train from test" | **False, and expensive.** 44 features are near-perfect train/test discriminators; dropping them costs **−0.0137** on the tail. A feature can be a perfect discriminator and still earn its place. |
| 4 | "normalise amount against a trailing global quantile" | Already implemented as `amt_rank_7d` / `amt_rank_30d`, retained, and inside the block that measures as essential. Not a new idea to spend on. |
| — | "reweight training rows toward the test distribution" | Ruled out by the shift diagnosis above, before any fitting. |

### The one change worth carrying forward

**A feedback delay on the target encoder.** A fraud label does not exist when
the transaction happens. Without a delay, a training row reads a perfectly
fresh encoder while a test row reads one frozen at the cutoff and up to 62 days
stale, and the model over-trusts it.

| delay | primary_62d | tail_late | tail_recent | wf_mar | wf_apr | wf_may |
|---|---|---|---|---|---|---|
| **7 d** | 0.7336 | 0.5481 | 0.5622 | 0.8170 | 0.7894 | 0.7969 |
| none | −0.0154 | −0.0080 | −0.0065 | −0.0182 | −0.0277 | −0.0190 |
| 3 d / 14 d / 30 d | ±0.001 | ±0.001 | ±0.001 | ±0.001 | ±0.003 | ±0.001 |

Removing it costs **6 of 6 windows**. Any delay from 3 to 30 days is
equivalent, so the effect is *having* one, not tuning one.
`config.TE_FEEDBACK_DELAY_D = 7.0`.

### The Stage 1 features are a no-result, and are recorded as such

42 new columns — 30-day RFM windows, within-customer amount percentile,
round-number and cents flags, per-customer category entropy and surprise,
duplicate counters, location-switch rates, account-youth interactions. All
leakage-clean, all cheap. Removing every one of them moves the tail by
**−0.0012, inside the noise floor.**

Not for lack of standalone signal: `amt_ratio_mean_30d` scores AP 0.183 on the
tail alone (11.5× base), second only to raw amount. It is redundant with
`amt_ratio_median` and `amt_rank_30d`, which the model already had. Some pieces
have no signal at all — round-number amounts sit at 1.00× lift, and exact
`(customer, merchant, amount)` repeats inside an hour **never happen** in this
data because amounts are continuous to the paisa.

They are kept for ensemble diversity, not claimed as an improvement.

### A worked example of the mistake this log exists to prevent

Two sub-threshold positives (drop the `graph` block, +0.0020; drop the new
amount columns, +0.0022) were combined, and the trimmed sets cleared the paired
test on the tail: +0.0033 and +0.0045, both significant.

Then they were checked on windows they had **not** been selected on:

| candidate | tail_late (selected on) | tail_recent | wf_mar | wf_apr | wf_may |
|---|---|---|---|---|---|
| drop graph + new amount | **+0.0033** | −0.0022 | +0.0003 | +0.0006 | −0.0027 |
| significant blocks only | **+0.0045** | −0.0035 | −0.0004 | +0.0001 | −0.0009 |

`tail_recent` scores the *identical rows* and disagrees in sign. The gain was
selection bias — the feature set was chosen by reading `tail_late`, so it fits
`tail_late`. **Neither trim adopted.**

This is the same failure as chasing the public leaderboard, at a smaller scale,
and the tail window is small enough to be very vulnerable to it. Anything
selected on the tail must be confirmed on `tail_recent` or the walk-forward
folds before it is believed.

---

## Submission 2 — the Stage 0/1 pipeline, and the calibration it bought

**`submissions/s2_pipeline_equal_lgb.csv`** — 243 features, 7-day encoder
feedback delay, corrected disjoint early-stopping window, equal-weight blend of
five LightGBM configs, two seeds each.

### Result: 0.54151 (public), up from 0.52708

The point of this submission was never the +0.014. It was the **second point on
the local↔leaderboard line**, and with two points there is finally a slope:

| | local `tail_late` | public LB | LB − local |
|---|---|---|---|
| submission 1 | 0.516 | 0.52708 | +0.011 |
| submission 2 | 0.5507 | 0.54151 | −0.009 |
| **change** | **+0.035** | **+0.014** | |

**A local gain arrives on the leaderboard at roughly 40% of its size.** Two
points is a thin line and the offset is not even constant in sign, so this is a
rule of thumb, not a calibration curve. But it is enough to reprice every
remaining candidate, and the repricing is severe: the entire Stage 2 search
produced nothing larger than 0.005 locally, which maps to **+0.002 on the
leaderboard — below the 0.003 noise floor.** Nothing in Stage 2 earns an upload
on expected score.

That is a result, not a failure. It says the pipeline is at the point where
feature engineering has been spent and fitting-procedure tuning has nothing left
to give, and the honest move is to stop buying lottery tickets with a budget of
ten.

### One more thing the number tells us

The leaderboard (0.5415) sits between `tail_late` (0.5509, 46–62 days ahead) and
`tail_far` (0.5304, 75–91 days ahead) — and close to their mean, 0.5407. That is
what you would expect if the test window's average drift exposure falls between
the two folds, which geometrically it does: the test spans 1–62 days past its
cutoff and then keeps drifting for another two months of calendar time. One
coincidence is not a finding, but if it holds on submission 3 it means the
proxy to select the final two submissions on is **mean(`tail_late`,
`tail_far`)**, not `tail_late` alone.

---

## Stage 2 and 3 — what the search found, and what it cost

Run as `tune.py` (fitting procedure) → `blend.py` (weights, post-processing) →
`submit.py` (refit, write). Selection on `tail_late`, confirmation on
`tail_recent`, and a candidate that helps one while hurting the other is
recorded as SPLIT rather than adopted.

### The search is a null result, and that is the finding

| question | candidates | best delta on `tail_late` | verdict |
|---|---|---|---|
| hyperparameters | 5 configs across the recommended ranges | −0.0060 to +0.0000 | nothing clears the floor |
| recency weighting | half-lives 7/14/21/45/90 d | +0.0026 | nothing clears the floor |
| window truncation | last 45/90/120 days only | +0.0006 | nothing clears the floor |
| model family | XGBoost ×2, CatBoost | +0.0020 | nothing clears the floor |
| entity post-processing | 3 entities × 4 statistics × 6 alphas | +0.0004 causal | nothing clears the floor |

Total spread across 16 fitting procedures: **0.008 AP**, on a window whose own
95% band is ±0.027. After 2.3–2.4× from feature engineering, the fitting
procedure has nothing left to give.

### Three things inside the null that are worth keeping

**Recency weighting is settled, and now we know why.** This was the repo's
largest open question, previously judged on the one fold that structurally
cannot see it. Judged on `tail_recent`, whose training data ends *inside* the
new regime, every half-life and every window is a no-result. The explanation:
**`win_45d` trains on 175,571 rows — 24% of the labelled data — and scores the
same as training on all 731,942.** Three quarters of the training set
contributes nothing measurable, so a scheme for down-weighting stale rows has
nothing left to correct.

**CatBoost was written off on stale evidence.** "XGBoost and CatBoost both score
below every LightGBM config" was true of the 201-column set on `primary_62d`. On
243 features, `cat_base` is the **best single model in the project** — 0.5530 on
`tail_late`, 0.7375 on `primary_62d` against the 0.7239 that the exclusion was
written about. Not a reason to switch; a reason the standing rule against
cross-family blending no longer has evidence behind it.

**Entity post-processing works only when it cheats.** Shrinking toward the
entity *mean* — the literal IEEE-CIS move — costs −0.09 to −0.28 AP, because
fraud here is per-transaction and pooling within a customer deletes the ordering
AP is computed from. Shrinking toward the entity *max* measured +0.0040 with a
paired interval of [+0.0019, +0.0063], and was adopted — then recomputed as
`cummax`, restricted to each entity's *earlier* rows. The gain fell to +0.0004.
The entire effect was a July test row being adjusted by a September one, which
is the two-sided aggregation the rules exclude. Rejected, and `blend.py` will
not adopt a non-causal variant at any score.

### Two defects found and fixed, one of which invalidated a claim

**The early-stopping window contained the window it was judging.** `tail_late`
(06-29 → 07-15) is a strict subset of `primary_62d` (05-15 → 07-15), so the
"stop on the widest window, report the narrow ones as free reads" convention was
choosing the round count from 1,068 of the tail's own positives.
`validation.stopping_fold` now returns a registered window disjoint from every
tail window and asserts it. Measured impact: **0.0002 AP** — the defect was real
and its effect was negligible, which is worth recording in both directions.

**The blend ladder adopted the most overfitting-prone option.** The first
version started from the simplest blend and adopted anything beating it by more
than the noise floor. Over 16 members it selected the hill-climb — which had put
weight on `hl_7d`, individually the *worst* member in the pool. A paired
bootstrap cannot see selection bias in weights fitted to the rows it resamples.
Rewritten: pick the best **unweighted** option outright, and require a fitted
blend to beat *that* by twice the floor. The hill-climb beats `equal_core` by
+0.0036 and the best unweighted option by +0.0008. Rejected.

---

## Submissions 3, 4 and 5 — how to read the results

All three were built after the pass-through was measured, so none of them is
expected to move the board much. Each is spent on a question whose answer
changes what the **final two** submissions should be, which is where the prize
actually is.

### 3 — `s3_equal_lgb_all`: does more averaging help? **No: 0.53901**

Thirteen LightGBM members equally weighted (5 hyperparameter configs + 8
recency/window variants), 2 seeds each. Local `tail_late` 0.5530, **+0.0023 over
submission 2**. On the board: **0.53901, −0.0025 under submission 2.**

**Local and leaderboard disagreed in sign.** This is the most useful negative
result of the day, and it is not "the gain was smaller than hoped" — it is that
the tail window had no information about this change at all. Eight models that
are the same LightGBM with different sample weights are near-clones; averaging
them adds no independent information and dilutes the members that were carrying
the ranking. The tail could not see that, because at 1,068 positives a 0.002
difference is indistinguishable from a reshuffle.

Adopted consequence: **the final blend stays small**, and no member is added
without a mechanism for why it makes *different* errors. It is also simpler to
reproduce for review, which matters at top-15.

### 4 — `s4_signup_integrity`: what is the artefact feature worth?

Submission 2 with `USE_SIGNUP_INCONSISTENCY=True` as the **only** change: same
five configs, same equal weights, same two seeds, same round book, 244 features
instead of 243. No re-tune, deliberately — a fresh round-count search would move
a second variable and make the answer unattributable. Head overlap with
submission 2 is 96.7%.

The flag was restored to `False` immediately after the fit and the feature-source
hash confirms the main line is back on the 243-column matrix.

**Result: 0.54874 — the best of the four, +0.0072 over submission 2.** Local
said ~+0.025, the prediction at 40% pass-through was ~+0.010, and the board gave
+0.0072. The feature is real and it survives into the test period.

So the price is now known, and the decision to leave it out of the main line is
a **judgement about the spirit of the rules made with the number in hand**,
which is the only honest way to make it. The rules say reverse-engineering the
generative assumptions "is not the intended path to a good score", and the top
15 face reproducibility review. `USE_SIGNUP_INCONSISTENCY` was restored to
`False` immediately after the fit and the feature-source hash confirms the main
line is back on the 243-column matrix.

### 5 — `s5_signup_plus_cat`: one different member instead of eight clones

The plan for this slot was `equal_all` — all 16 members across three families.
**Submission 3's result cancelled it mid-build.** If padding a blend with eight
near-clone LightGBM variants costs 0.0025 on the board, a 16-member blend
containing those same eight is not a good bet, whatever the local tie says. The
job was killed with CatBoost already fitted and cached.

What replaced it is built on the two things the board actually established
today: submission 4's configuration is the strongest measured (0.54874), and
members should be added for *decorrelation*, not for count. So: the five
LightGBM configs on the 244-column matrix, plus `cat_base` — one member, a
different algorithm, different failure modes, and the best single model in the
project (`tail_late` 0.5530, `primary_62d` 0.7375).

A clean-line version, `s5_core_plus_cat`, was built first and is kept in
`submissions/`. It was not uploaded: its head overlap with submission 2 is
**98.7%**, so it is a near-duplicate with an expected gain around +0.001. That
is the honest state of the clean line — **there was no clean change left today
with a meaningful expected gain**, and the only lever with a measured effect was
the one deliberately scoped to a single measurement. Using it a second time was
an explicit decision, taken with the +0.0072 price already known, and it does
not make the feature the pipeline default.

**Result: 0.55014 — the day's best, +0.0014 over submission 4.**

And it says something submission 3 did not. Both were "add members to a blend",
both had local deltas far below the 0.008 resolution limit, and they behaved
completely differently:

| change | members added | local Δ | board Δ |
|---|---|---|---|
| submission 3 | 8 LightGBM recency/window variants | +0.0023 | **−0.0025** |
| submission 5 | 1 CatBoost | +0.0011 | **+0.0014** |

The distinguishing property is not the size of the local delta — submission 3's
was larger. It is whether the added members are *independent*. Eight LightGBMs
differing only in sample weights make the same mistakes, so averaging them
dilutes without informing. One CatBoost, with ordered boosting and a different
regularisation path, makes different mistakes. **Add members for decorrelation,
not for count** — and note that the local estimate tracked the board correctly
for the decorrelated addition and inverted for the clones, which is a better
diagnostic than either number alone.

Hold the +0.0014 loosely: it is a public-60% figure on two files sharing 98.9%
of their top 1%, and it does not guarantee the private 40% agrees.

---

## Day 2 — the search closes, and the reproducibility hole opens

Day 2 opened at 0.55014 and rank 14 of a leaderboard led by 0.57. Five
submissions left, two private slots, and the round closing the same night. The
day was planned around three questions: is there a local statistic that resolves
below 0.008, is there an ensemble member worth adding, and is the leading
submission actually reproducible. The answers are no, no, and no — and the third
one turned out to be the only one that mattered.

### The far horizon is now filled in, and it earns its keep

`tune.py --far` was run over every candidate so all 20 members carry
`tail_recent` (1–17 d), `tail_late` (46–62 d) and `tail_far` (75–91 d) on one
matrix. Two things follow immediately.

**The ordering genuinely changes with horizon.** `lgb_extra` is 7th of 8 at
`tail_late` and **1st at `tail_far`**; `lgb_deep` is 2nd at `tail_recent` and
last at `tail_far`. A single-horizon read is not just noisy, it is answering a
different question than the private half of the test set asks.

**Blend decay rises monotonically with member count**, which finally gives
submission 3's board loss a mechanism rather than a story:

| composition | members | tail_late | tail_far | decay |
|---|---|---|---|---|
| `equal_core` (= s2) | 5 | 0.5511 | 0.5342 | **0.0169** |
| `equal_lgb_all` (= s3) | 13 | 0.5532 | 0.5335 | **0.0197** |
| `equal_all` | 20 | 0.5517 | 0.5328 | 0.0190 |

The thirteen-member blend is *better* near and *worse* far. It bought
near-horizon AP and paid for it in the regime the private 40% sits in, which is
exactly the trade the board charged 0.0025 for.

### `mean(tail_late, tail_far)` is a better estimator and still not a discriminator

The hypothesis was that the board lands between the two horizons, so their mean
should predict it. As a **level** estimator that is clearly right:

| composition | tail_late | mean(late,far) | board | err(late) | err(mean) |
|---|---|---|---|---|---|
| `equal_core` | 0.5511 | 0.5427 | 0.5415 | +0.0096 | **+0.0012** |
| `equal_lgb_all` | 0.5532 | 0.5433 | 0.5390 | +0.0141 | **+0.0043** |

As a **discriminator** it fails the same way `tail_late` does: the board puts s2
above s3 by 0.0025, and `mean(late,far)` puts s3 above s2 by 0.0007. So the
0.008 resolution limit is not an artefact of a badly-centred statistic — 1,068
positives cannot resolve 0.0025 no matter what functional is computed from them.
That is a stronger and more useful version of the standing rule.

(`tail_far` *alone* happens to get the sign right. With three candidate
statistics and one comparison, one agreeing by chance is the expected outcome,
and it is recorded here specifically so it does not get promoted to a rule.)

### Four members proposed with mechanisms; three refuted, one a clone

Each was specified in advance with a reason it should make *different* errors,
then judged on a four-part gate — mechanism, decorrelation (rank ρ ≤ 0.70
against the blend), competence (within 0.010 of the core at `tail_late`), and
horizon (decay no worse than the core's).

| member | mechanism | ρ | tail_late | tail_far | verdict |
|---|---|---|---|---|---|
| `lgb_durable` | no `amount` block; that family retains only 0.61–0.64 of its power late | 0.777 | 0.5397 | 0.5197 | **refuted** — decay 0.0200, *worse* |
| `cat_durable` | same view, decorrelating family | 0.632 | 0.5385 | 0.5143 | **refuted** — decay 0.0242 |
| `lgb_linear` | linear leaves extrapolate past split points | 0.572 | 0.5484 | **0.5016** | **refuted** — decay 0.0468 |
| `lgb_sub30` | feature bagging over a redundant space | 0.829 | 0.5483 | 0.5331 | competent, **is a clone** |

**`lgb_durable` is the instructive failure.** Per-feature retention says the
amount family loses 36–39% of its standalone power across the regime split while
velocity and graph lose 1–11%, so a model denied `amount` should decay more
slowly. It decays *faster* (0.0200 vs 0.0169) and is 0.0094 worse at `late`.
Standalone feature retention does not predict how a model built on those
features behaves — the same gap between gain share and marginal value the repo
already documents for `encoding` vs `behaviour`, in a new place.

**`lgb_linear` is why the far read exists.** It ties the core on the selection
target (0.5484 vs 0.5491), passes the decorrelation gate comfortably at ρ 0.572,
and has a documented mechanism. It would have passed every check this project
had before today. At 75–91 days it collapses to 0.5016 — a decay of 0.0468,
nearly 3× anything else measured. The literature's caveat fits better than the
hypothesis did: linear leaves extrapolate as linear functions and *diverge*, and
the unbounded counters are far outside their training range by then. (It also
early-stopped at 60 rounds, so under-training is a competing explanation; it
fails either way.)

**Decorrelation and competence trade off against each other here.** The two most
independent members in the project (`lgb_linear` 0.572, `cat_durable` 0.632) are
precisely the two that fall apart at the far horizon; the one that matches the
core everywhere is a clone at 0.829. `cat_base` — ρ 0.624 *and* competitive at
all three horizons — is the sole exception, which is why it was the member that
paid, and it now looks like a rare object rather than one draw from a family of
possible additions. Note also that `xgb_base` sits at ρ 0.824, inside the
LightGBM clone band: **a different library is not automatically a different
model.**

### Nothing improves the blend, including the weights

Equal weight per *family* rather than per member (CatBoost at ½ instead of ⅙)
measures +0.0007 on `tail_late` with a paired interval of [−0.0003, +0.0018] —
inside the floor, interval spanning zero. Adding `xgb_base` moves it −0.0001.
The s5 composition is not improvable with the members that exist.

### The second private slot is worth ~0.00006

Measured on the actual submission CSVs, every candidate pair shares **93.6–99.0%
of its top 1%**, and average precision reads the head of the ranking. Dropping
all the way to a single model (`cat_base` alone) only gets head overlap down to
94.2% against the blend, at a cost of 0.0012 on `mean(late,far)`.

Working the max-of-two arithmetic: the paired interval between those two gives
σ ≈ 0.0013 on the tail, ≈0.0010 scaled to the private draw's ~1,786 positives,
against a 0.0012 expected deficit. `E[max] − E[best]` ≈ **+0.00006 AP**. The
second slot cannot be made to do useful work, and manufacturing a "hedge" would
only cost expected score. Select the two highest expected scores and say so.

### The finding that changed the day: the pipeline is not reproducible

`lgb_deep` early-stopped at **230 rounds** on day 1 and **758** on day 2 — same
code, same data, same params, same disjoint stopping window — for a `tail_late`
change of 0.0011. The cause is not LightGBM: `lgb_base` reproduced to four
decimals across three separate processes today. It is the feature matrix.

| gnn column | bit-identical across builds | max abs diff | correlation |
|---|---|---|---|
| `gnn_cd_score` | no | 3.2e-04 | 1.000000 |
| `gnn_cd_cos` | no | 1.5e-05 | 1.000000 |
| `gnn_cm_cos` | no | 1.0e-05 | 1.000000 |
| `gnn_cust_emb_drift` | no | 3.0e-06 | 1.000000 |

`gnn.py` **is** seeded — `torch.manual_seed`, seeded generators, a seeded state
initialiser. This is float non-determinism in multi-threaded CPU reductions. A
3e-4 perturbation of four columns out of 244 is enough to move an early stop by
3×, which says the AP stopping curve is a plateau, not a peak.

Three consequences, and the first two are now enforced in code:

1. **Round counts must be pinned, not re-derived.** `submit.py --rounds-book`
   takes the book a submission was built from;
   `tune_rounds.s5_asbuilt.json` preserves the counts that produced 0.55014.
   Naming the same members is *not* enough to identify the same model.
2. **The matrix must be shipped, not just the code.** The 244-column build that
   produced submissions 6 and 7 is preserved at
   `data/processed/base_features.signup.parquet`.
3. **A paired-bootstrap verdict is not stable across a rebuild.**
   `lgb_shallow` measured −0.0007 against `lgb_base` on day 1 and **+0.0032,
   verdict `BETTER`, interval excluding zero** on day 2. The verdict machinery
   promoted a candidate on nothing but early-stopping churn. This is the
   sharpest evidence yet for the repo's own rule that sub-0.008 deltas are not
   results — the round book is a *larger* run-to-run variance source than the
   0.003 seed noise that was documented.

This mattered because **s5 could not be reproduced.** It was fitted with
`deterministic=False` at 2 seeds, and its feature matrix was overwritten. At
rank 14 — on the top-15 boundary, where rulebook 8.2 forfeits a position that
cannot be reproduced from the submitted notebook — that is the largest
uninsured risk on the board, and it is not a modelling problem.

### 6 — `s6_det_s5comp`: the reproducible finalist

Submission 5's composition exactly (five LightGBM configs + `cat_base`, 244
columns), refit with `--deterministic` across all three families at 3 seeds,
round counts pinned to the as-built book. Head overlap with s5 is **99.0%**, so
the expected board delta is ~0.

That is the point. It is spent on reproducibility, not score. A *large* move
here would itself be a finding — it would mean the run-to-run floor is wider
than the documented 0.003.

### 7 — `s7_family_equal`: the only remaining structural choice

The same six members weighted equally per family rather than per member. Local
+0.0007, below the floor; taken because an unspent upload is worth nothing and
this is the highest-expected-score candidate left, **not** because it hedges —
its head overlap with s6 is 97.9%.

### What 6 and 7 returned, and why the pair is worth more than either score

| | composition | seeds / det | public LB |
|---|---|---|---|
| s5 | member-equal, 6 members | 2, non-det | 0.55014 |
| s6 | **identical composition**, pinned rounds | 3, det | **0.54995** |
| s7 | same members, family-equal weights | 3, det | **0.55150** |

**s5 → s6 is the cleanest measurement this project has made.** Same members,
same pinned round counts, same feature matrix; only the seed count and
determinism changed. The board moved **0.00019**. Every "noise floor" figure in
this repo — 0.003, inherited from a comparison that also rebuilt the features —
is an order of magnitude too conservative *for a same-composition rebuild*. The
board can resolve differences this repo has been treating as unresolvable.

**That makes s6 → s7 readable.** The two differ in nothing but the weight
vector, and the board moved **+0.00155** — about 8× the floor just measured, in
the direction local predicted (+0.0007 on `tail_late`, interval
[−0.0003, +0.0018]). Local and board agreed in sign and roughly in size, which
they have not done for a change this small before.

Two corrections to standing beliefs follow. The **noise floor is not one
number** — it depends on what varied. Rebuilding the features moves a round
count 3× and the score by ~0.002; holding the matrix fixed and changing seeds
moves it by 0.0002. And **weighting is not "post-processing that does not
matter"**: shifting half the blend's mass onto the one decorrelated member is
the second-largest per-submission gain of the whole competition after the
feature rebuild and the integrity feature.

### Where that leaves the last three submissions

The local CatBoost-weight curve is flat where it matters:

| w_cat | 0.167 (s6) | 0.400 | **0.500 (s7)** | 0.600 | 0.750 | 1.000 |
|---|---|---|---|---|---|---|
| `tail_late` | 0.5519 | 0.5524 | **0.5526** | 0.5526 | 0.5526 | 0.5516 |
| mean(late,far) | 0.5434 | 0.5440 | **0.5441** | 0.5441 | 0.5439 | 0.5422 |

s7 sits on the peak, and six a-priori variants of the *core* side — dropping the
most redundant member (ρ 0.915), dropping the steepest-decay member, adding
XGBoost at 0.1 — all land within 0.0002 of it. **There is no local signal left
to follow**, and following the board alone at this scale is the public-LB
chasing the rules explicitly warn against.

One mechanism remains that the weight curve structurally cannot test. All of the
decorrelated half currently rides on a *single* model (seed-averaged, but one
configuration). Splitting that half across genuinely different CatBoost
configurations — `cat_deep` (depth 10) and `cat_rsm` (random subspace, Bernoulli
bootstrap) — reduces the decorrelated half's own variance without shrinking its
weight. That is the last question worth a submission.

### 8 — `s8_clean_line`: pricing the integrity feature on the model actually shipped

s7's exact composition, weights and round book on the 243-column clean matrix;
the single variable is `signup_inconsistency_d`. **0.54548, so the feature is
worth +0.00602 here** — against the +0.0072 measured by submissions 4 vs 2 on
the older member-equal blend. Consistent, and now priced on the model being
submitted rather than inherited from a different one.

It is also the most decorrelated file in the set (top-1% overlap 0.970 with s7,
against 0.98–0.99 for every other pair), because removing a feature changes the
model in a way reweighting one cannot. Not a contender at 0.006 behind, but a
documented fallback if the feature is ever questioned.

### 9 and 10 — the weight curve has no interior maximum

Local's curve is flat from w_cat 0.4 to 0.75 and turns down after; with the
refreshed CatBoost predictions it peaks at 0.65 and puts 0.80 *below* 0.50. The
board disagrees:

| w_cat | 0.167 | 0.500 | 0.650 | 0.800 |
|---|---|---|---|---|
| **public LB** | 0.54995 | 0.55150 | 0.55187 | **0.55203** |
| step | — | +0.00155 | +0.00037 | +0.00016 |
| local mean(late,far) | 0.5434 | 0.5447 | **0.5448** | 0.5445 |

Monotone increasing and sharply decelerating — **+0.00208 in total from moving
weight onto the one decorrelated member.** After the feature rebuild (+0.0144)
and the integrity feature (+0.0060), this is the third-largest effect of the
competition, and it came from a parameter that costs nothing to change.

Three things worth keeping, and one admission.

**The prediction was wrong.** s10 was uploaded expecting a *decline*, to
demonstrate an interior optimum for the write-up. There isn't one inside the
tested range. Recorded because a plan that survives its own test teaches less
than one that does not.

**The last step is below the floor.** +0.00016 at w = 0.65 → 0.80 sits under the
0.00019 same-composition floor measured at s5 → s6, so that increment alone is
not distinguishable from noise. Four monotone points are evidence; the fourth
step on its own is not.

**Local and the board disagree about *where* the optimum is, exactly as the
resolution limit predicts.** Local says 0.65, the board says ≥0.80, and the gap
between them (0.0003) is far under the ~0.008 the tail window can resolve.
Neither is "wrong" — the question is finer than the local instrument.

**What was ruled out, with reasons, before settling on the weight probes:**

| candidate | why not |
|---|---|
| splitting the CatBoost half across `cat_base` + `cat_rsm` (+ `cat_deep`) | exact tie at w = 0.50, 0.65 **and** 0.80 — the "more weight makes variance matter more" argument is wrong |
| CatBoost `boosting_type="Ordered"` | its benefit is avoiding leakage in *categorical* target statistics, and this matrix is entirely numeric. Timed anyway: **12–13× slower** than Plain, ≈2 h for one fit at 492k rows — infeasible, and for an inactive mechanism |
| mixing in the clean line (s8) | 0.006 behind at 97% head overlap; cannot pay for itself |
| more seeds | s5 → s6 measured this at −0.00019 |

### The private pair

**`s10_cat80` (0.55203) + `s9_cat65` (0.55187).** They are simultaneously the two
highest public scores *and* the two candidates that bracket the local-vs-board
disagreement about where the weight optimum sits — local's peak is s9's 0.65,
the board's is s10's 0.80. Both are deterministic, pinned to
`tune_rounds.s5_asbuilt.json`, built on the preserved 244-column matrix, and
therefore reproducible from the notebook.

`s5_signup_plus_cat` (0.55014) is deliberately **not** selected despite beating
s6: it was fitted non-deterministically at 2 seeds and its feature matrix has
been overwritten, so it is the one submission that cannot satisfy rulebook 8.2.

**Final: 0.52708 → 0.55203 across ten submissions, +0.02495.** Roughly 58% of
that is the Stage 0/1 feature and validation rebuild, 24% the integrity feature,
and 8% the blend weighting — with the remainder spread across the decorrelated
CatBoost member and seed/determinism changes.

---

## Revised plan for the remaining submissions

Everything below is priced through the measured pass-through, which after four
submissions has a **threshold** in it rather than a single slope:

| local delta | board delta | pass-through |
|---|---|---|
| +0.035 (sub 2) | +0.0144 | 41% |
| +0.025 (sub 4) | +0.0072 | 29% |
| +0.0023 (sub 3) | **−0.0025** | **wrong sign** |

**`tail_late` is a sound proxy above roughly 0.008 AP and has no resolving power
below it.** That is a stronger claim than "small differences are noisy": a
sub-0.005 local delta is not a small board gain, it is a coin flip. 67,196 rows
and 1,068 positives cannot separate models that close, and no amount of
bootstrapping fixes a resolution limit. The entire Stage 2 search lives below
that threshold, which is why none of it was adopted.

The competition's own advice applies and now has numbers behind it: *"Do not
chase the public leaderboard... Trust a robust, time-aware local validation
setup over repeated public submissions."*

All four of today's uploads are spent and scored:

| # | file | question it was spent on | public LB |
|---|---|---|---|
| 2 | `s2_pipeline_equal_lgb` | does the Stage 0/1 rebuild move the board? | 0.54151 |
| 3 | `s3_equal_lgb_all` | does averaging 13 differently-memoried models help? | 0.53901 |
| 4 | `s4_signup_integrity` | what is `signup_inconsistency_d` worth? | 0.54874 |
| 5 | `s5_signup_plus_cat` | does one decorrelated member beat eight clones? | **0.55014** |

**0.52708 -> 0.55014 in one session, +0.0231.** Roughly two thirds of that is
the Stage 0/1 feature and validation work (+0.0144) and one third the integrity
feature plus CatBoost (+0.0086).

### The private pair -- five submissions left, decide before the deadline

The board is 60% public / 40% private, and **at most two submissions count**.
The choice is not "the two highest public scores"; it is a hedge against the
things that could differ on the private 40%.

| candidate | public | case for it |
|---|---|---|
| `s5_signup_plus_cat` | 0.55014 | highest public; if the private split behaves like the public one, this wins |
| `s2_pipeline_equal_lgb` | 0.54151 | the **clean line** -- no generation-artefact feature. Insurance against `signup_inconsistency_d` failing on unseen months *or* being questioned at reproducibility review |
| `s4_signup_integrity` | 0.54874 | maximises expected score if the artefact holds, but shares 98.9% of its top 1% with `s5` -- a poor hedge, it fails in the same way |

The default (selecting nothing) takes the best public score, which is `s5`.
Choosing `s5` + `s2` costs ~0.009 of expected public score and buys real
diversification, because the two differ in the one feature whose behaviour on
unseen months is least certain. Choosing `s5` + `s4` is not a hedge at all.

**This is a judgement call, not a computation, and it should be made
deliberately rather than by letting the default fire.**

**Do not spend a submission on:** dropping the amount block, dropping
drift-flagged features, importance weighting, `scale_pos_weight`, either trimmed
feature set, recency weighting, window truncation, hyperparameter variants,
entity post-processing, or padding the blend with more LightGBM variants. All
now have answers, and the last one has a *leaderboard* answer.

**The only kind of ensemble member worth adding** is one that makes independent
errors — a different algorithm, not a reweighting of the same one. That is the
single actionable modelling lesson of the day: worth +0.0014 where eight extra
LightGBMs were worth −0.0025.

**Still to try locally, before anything earns another submission:** fill the
`tail_far` reads for the params and family sets so the chosen blend has a
75–91-day score — the leaderboard landed between `tail_late` and `tail_far` and
close to their mean, and if that survives submission 3 it is the proxy the final
two picks should be made on. Then: `te_merchant_id` as a single-column ablation;
`--deterministic` for anything that could face reproducibility review; and
replacing the 44 unbounded counters with rate/share forms, since deleting them
is known to hurt.

---

## Standing lessons

### Do

- **Score every candidate on the tail window** (2026-06-29 → 2026-07-15), not
  just on `primary_62d`. Report both; if they disagree, believe the tail.
- **Report AP as lift over the slice's own base rate** as well as raw AP.
  Base rates vary 0.0139–0.0193 week to week, and raw AP moves with them.
- **Prefer features that held their power late** — velocity, recency, graph —
  when trading off against amount-family features.
- **Spend each submission on a question**, and make the change large enough that
  the answer survives the noise floor. As of submission 2 that threshold is
  measurable: local gains arrive on the leaderboard at roughly 40% of their
  size, so a change worth less than ~0.008 locally cannot clear the 0.003 board
  noise and is not worth an upload.
- **Check the head overlap before spending an upload.** Average precision reads
  mostly the top of the ranking. Submission 2 had rank correlation 0.80 with
  submission 1 and still shared **94.6% of its top 1%** — two files that look
  different overall can be near-identical where the metric looks. `submit.py`
  prints this against the previous submissions automatically.
- **Separate unweighted blends from fitted ones.** An equal average over an
  a-priori group learns nothing from the evaluation window and can be adopted on
  its measured score. A hill-climb fits its weights to the same rows the
  bootstrap resamples, so its interval understates its own selection bias — hold
  it to twice the noise floor against the best unweighted option.

### Don't

- **Don't read differences below ~0.003 AP as signal.** LightGBM runs with
  `num_threads=0` and without `deterministic=True`, so two runs of identical
  code differ: top-500 overlap 98.4%, top-1% Spearman 0.9989, bottom-90%
  Spearman 0.9595. The metric only looks at the head, so this is harmless for
  scoring — but it means small LB deltas are not evidence.
- **Don't average AP over a window that spans a regime change.** That is the
  single mistake that produced a 0.20 surprise here.
- **Don't trust a "drift-robust" feature without measuring it under the actual
  drift.** Every relative amount form here was designed for robustness and lost
  ~38% of its power anyway.
- **Don't chase the public LB with near-duplicate submissions.** It is 60% of
  the test set and the private 40% is a different draw; the rules warn about
  this explicitly.
- **Don't flip `USE_SIGNUP_INCONSISTENCY` without an explicit decision.** It is
  worth ~+0.025 AP and is not target leakage, but it is a fingerprint of how the
  fraud rows were synthesised, and top-15 teams face reproducibility review.
  Submission 4 measures it once, on an explicit decision, and does **not** adopt
  it for the main line.
- **Don't post-process predictions across the whole scored set.** Grouping test
  rows by entity and taking a mean or max looks like ordinary post-processing
  and is not: it lets a July row be adjusted by a September one. The rules allow
  aggregation over test rows only "in a strictly-past-only way". Here the
  two-sided form measured +0.0040 and the causal form +0.0004 — the entire
  apparent gain was the lookahead.
- **Don't assume an early-stopping window is independent of what it stops for.**
  `tail_late` is a strict subset of `primary_62d`, so "stop on the widest window
  and report the narrow ones as free reads" was silently choosing the round
  count from 28% of the reported window's own positives.

### Non-negotiable (rule compliance)

No external data. No feature built from any fraud label, including past labels.
No `transaction_id` or raw epoch time in the model matrix. No random or
stratified K-fold anywhere. No `fit` of any model, encoder, scaler or target
statistic on test rows.

---

## Plan for the remaining 9

> **Superseded.** See "Revised plan for the remaining 9" above, which reflects
> what Stage 0 and Stage 1 settled locally. The original table below is kept
> because two of its four hypotheses were answered *negatively*, and the record
> of a plan that did not survive contact with measurement is worth as much as
> the plan that did.

The finding above points at one dominant lever, so the next submissions should
test it in order of expected value, largest change first.

| # | hypothesis | change | expected read |
|---|---|---|---|
| 2 | Recent data is far more representative than old data | Retrain the blend with an aggressive recency half-life (**7–21 d**, vs the 45/90 d already tried) and select on the **tail window** | If LB moves +0.02 or more, drift-adaptation is the whole game and submissions 3–5 tune it |
| 3 | The amount family is actively harmful late | Down-weight or drop the most drift-exposed amount features; lean on velocity / entity / graph | Isolates how much of the loss is amount-specific |
| 4 | The trend is smooth enough to extrapolate | Normalise amount against a *trailing global* quantile so the fraud archetype sits at a stable normalised position | Tests whether the drift can be modelled rather than merely survived |
| 5+ | — | reserve; decide from the results of 2–4 | keep ≥2 for a safe final pick |

**Reserve at least two submissions.** Two entries get selected for private
scoring, and the private 40% is a different sample — so the final pair should be
one aggressive candidate and one conservative one, not two variants of the same
idea.

### Still open from before

- `te_merchant_id` ranks 8th by gain (3.4%) but was never A/B'd against dropping
  it. The plan called for gating it on fold AP; that gate is untested.
- Final-model round counts are fold counts x 1.2 — a convention, not a validated
  choice.
- `deterministic=True` is off. Worth turning on for whichever submission would
  face reproducibility review.

---

## How to reproduce a logged submission

```bash
.venv/Scripts/python.exe build_features.py --check
.venv/Scripts/python.exe -u train_model.py --stage all
.venv/Scripts/python.exe calibrate_submission.py --inplace
```

Delete `data/processed/base_features.parquet` first if any feature module
changed, or the run will train on stale features.
