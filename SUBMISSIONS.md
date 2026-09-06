# Submission log — REACT 2026

Budget: **10 submissions total**, max 5/day. Leaderboard is **60% public / 40%
private**; up to **2 submissions** may be selected for private scoring (default:
best public). The rules say explicitly: *"Do not chase the public leaderboard…
Trust a robust, time-aware local validation setup over repeated public
submissions."*

So each entry below records not just the score but **what question the
submission was spent to answer**. A submission that answers nothing is wasted
even if it scores well.

| # | date | file / md5 | local val AP | public LB | Δ vs local | question it answered |
|---|---|---|---|---|---|---|
| 1 | 2026-09-06 | `submission.csv` `ab115128…` | 0.7252 (`primary_62d`) | **0.52708** | **−0.198** | Does local validation predict the leaderboard? **No — and we now know exactly why.** |
| 2 | — | — | — | — | — | — |
| 3 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — |
| 6 | — | — | — | — | — | — |
| 7 | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — |
| 9 | — | — | — | — | — | — |
| 10 | — | — | — | — | — | — |

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

## Standing lessons

### Do

- **Score every candidate on the tail window** (2026-06-29 → 2026-07-15), not
  just on `primary_62d`. Report both; if they disagree, believe the tail.
- **Report AP as lift over the slice's own base rate** as well as raw AP.
  Base rates vary 0.0139–0.0193 week to week, and raw AP moves with them.
- **Prefer features that held their power late** — velocity, recency, graph —
  when trading off against amount-family features.
- **Spend each submission on a question**, and make the change large enough that
  the answer survives the noise floor.

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

### Non-negotiable (rule compliance)

No external data. No feature built from any fraud label, including past labels.
No `transaction_id` or raw epoch time in the model matrix. No random or
stratified K-fold anywhere. No `fit` of any model, encoder, scaler or target
statistic on test rows.

---

## Plan for the remaining 9

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
