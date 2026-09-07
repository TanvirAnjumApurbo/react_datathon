# REACT 2026 — Method Summary

**Task:** per-transaction fraud probability for a 62-day forward block
(2026-07-16 → 2026-09-15) beginning the day after the labelled data ends.
**Metric:** PR-AUC (average precision). **Base rate:** 1.76%.

---

## 1. Approach in one paragraph

The organisers supply only raw transaction fields, so the work is reconstructing
behaviour — what is normal for this customer, device, merchant and location, and
how far this transaction departs from it — using only information strictly
earlier than the row being scored. We build 243 such features over the combined
train+test stream, prove the past-only property empirically rather than
asserting it, and feed them to a small ensemble of gradient-boosted trees. The
harder half of the problem turned out not to be the features but the
**validation**: fraud in this dataset drifts, and a conventional forward fold
overstated leaderboard performance by 0.20 AP.

## 2. Features (243 columns, seven blocks)

| block | n | what it captures |
|---|---|---|
| `velocity` | 57 | customer / device / merchant recency and rolling counts, 1h → 168h |
| `entity` | 55 | pair novelty, device sharing, diversity, location movement |
| `amount` | 40 | amount against the customer's own history; trailing population percentile ranks |
| `graph` | 29 | weekly snapshot bipartite graph: degree, components, spectral, learned embeddings |
| `behaviour` | 28 | habit/surprise, repetition, travel rate, account-youth interactions |
| `temporal` | 22 | hour/day, cyclical encodings, night flag, per-customer circular rhythm |
| `encoding` | 13 | past-only smoothed target encoding with a 7-day feedback delay |

Gain share: amount 48.6%, velocity 27.4%, entity 13.1%, graph 3.4%, temporal
2.9%, behaviour 2.4%, encoding 2.4%.

Three design decisions did most of the work.

**Relative beats absolute.** The two sharpest signals are relational, not
absolute: an amount more than 10× the customer's own past mean is 92.8% fraud
(52.7× lift), and a customer whose previous transaction was under a minute ago
is 86.0% fraud (48.9× lift). Merchant velocity, by contrast, is worthless
(0.69× lift) — the merchants are high-traffic aggregators, with the top 10
carrying 57.8% of all rows.

**A stale target encoder beats a fresh one.** A fraud label does not exist when
the transaction happens; it exists once an investigation confirms it. Giving the
encoder a 7-day feedback delay — so a training row reads labels as they stood a
week earlier — is worth 0.008–0.028 AP on *all six* validation windows. Without
it, training rows see a perfectly current encoder while test rows see one frozen
at the cutoff and up to 62 days stale, and the model calibrates its trust to a
freshness it will never get at scoring time.

**Rates and shares, not lifetime counts.** Anything monotone in calendar time
extrapolates off a cliff at the train/test boundary, so each raw entity age is
paired with a fraction-of-observable-history version. We verified this matters
and also verified the obvious over-correction fails: 44 features separate train
from test at up to AUC 0.997, and *dropping* them costs 0.0137 AP on the tail. A
feature can be a near-perfect train/test discriminator and still earn its place.

## 3. Leakage safety

**Structurally**, every historical statistic routes through a single primitive
(`features/_windows.py:WindowIndex`) that resolves each row's window start with
one global `searchsorted` over a composite `group × BIG + timestamp` key, so the
past-only guarantee lives in one place rather than being re-argued per feature.

**Empirically**, five guards run in the submitted notebook and fail loudly:
no banned columns or duplicates; no single feature individually near-perfect;
a customer's first-ever row has no history features; train/test boundary
continuity; and the decisive one — **truncation invariance**, which rebuilds the
whole pipeline on a stream cut at a cutoff and diffs it against full-stream
values on the shared rows. If any feature at time *t* used information after *t*,
deleting the future would change it.

The single-feature scan is worth quoting, because it is measured on the full
244-column matrix actually submitted — including the disclosed integrity feature
of §8 — and the top of it is unremarkable:

| feature | standalone AP |
|---|---|
| `amt_ratio_median` | 0.4400 |
| `log_amt_ratio_mean` | 0.4211 |
| `amt_ratio_mean` | 0.4208 |

A leaked column sits near 1.0. Nothing here is close, and the strongest signal
in the model is "this amount is unlike the customer's own history" — which is
the behavioural question the competition asks.

A sixth guard exists because the other five all passed while a real defect was
live: the weekly graph snapshots left 29 columns NaN for 10,455 **test** rows and
zero train rows, and since those rows carry no labels and sit in no fold, nothing
caught it. `check_tail_coverage` now compares each feature's NaN rate over the
last 7 days of the stream against its rate elsewhere.

No feature uses any fraud label, including past labels. No model, encoder or
target statistic is fitted on test rows. Validation is forward-block only.

## 4. Validation, and the mistake it exists to prevent

Our first submission scored **0.7252** on a 62-day forward fold and **0.52708**
on the leaderboard. Nothing was leaking — the fold averaged a regime change away.

Fraud amounts decay across the stream while legitimate behaviour does not: fraud
median amount falls 4,918 BDT (January) → 2,044 (May) → 944 (mid-July), while
the legitimate median holds at ~485 for all 37 weeks. Inside that 62-day fold
the weekly AP is flat near 0.79 for six weeks and then falls to 0.599 and 0.473,
and **those last two weeks average almost exactly the leaderboard score.** The
test period sits entirely inside the new regime and is still moving.

Adversarial validation shows this is **concept drift, not covariate shift**: a
train-vs-test classifier reaches AUC 0.612 on raw columns and 0.514 — chance —
once `account_age_days` is removed, and `amount_bdt` has PSI 0.0002 between the
periods. The amount *distribution* did not move; `P(fraud | amount)` did. That
ruled out density-ratio importance weighting on principle before we spent any
time fitting it.

Selection therefore moved to the last 2.5 labelled weeks, read at three forecast
horizons over the *same rows*: `tail_recent` (1–17 days ahead), `tail_late`
(46–62, matching the far end of the real test window) and `tail_far` (75–91, a
deliberate stress read). The horizon reorders the models — one configuration is
7th of 8 at 46–62 days and 1st at 75–91 — so a single-horizon read answers a
different question than the private half of the test set asks.

## 5. Model

Six members: five LightGBM configurations spanning the ranges the literature
recommends, plus CatBoost. Three seeds each, rank-averaged within a member, then
combined with equal weight. Ranks rather than probabilities, because average
precision reads only the ordering and the two families are on different scales.

**Ensemble members were added for decorrelation, not for count**, and we paid a
submission each way to learn it. Adding eight LightGBM variants that differed
only in sample weights gained +0.0023 locally and **lost 0.0025** on the
leaderboard; adding one CatBoost gained +0.0011 locally and **gained 0.0014**.
Measured as rank correlation against the LightGBM core, CatBoost sits at 0.62
while every LightGBM variant sits at 0.74–0.91 — the same band the core members
occupy among themselves. XGBoost sits at 0.82, inside that band: a different
library is not automatically a different model.

The same logic decides the weights. Equal weight per *member* gives the one
decorrelated model 1/6 of the blend; moving it to 1/2, then 0.65, then 0.80
gained 0.00155, 0.00037 and 0.00016 on the leaderboard — monotone, sharply
decelerating, and +0.00208 in total, against a same-composition noise floor we
measured at 0.00019. Our local window could not resolve this: it is flat from
0.4 to 0.75. It is the one place where the leaderboard was the finer instrument,
and it is finer only because the models being compared are near-identical, so
the paired comparison cancels almost all sampling noise. We later confirmed the mechanism: blend decay across
the forecast horizon rises monotonically with member count (0.0169 at five
members, 0.0197 at thirteen), so extra correlated members buy near-horizon
accuracy and pay for it at long horizons.

**What we deliberately did not do**, each measured rather than assumed:
no `scale_pos_weight` (unweighted wins 6 of 6 windows — AP reads only the
ordering, so upweighting positives adds no information and distorts leaf
values); no SMOTE or resampling; no recency weighting or training-window
truncation (half-lives 7–90 d and windows 45–120 d, measured on the fold built
to detect them, none clearing the noise floor); no fitted blend weights (a
hill-climb optimises on the same 1,068 positives the bootstrap resamples, so its
interval cannot see its own selection bias); and no entity post-processing —
shrinking each score toward its entity's max measured +0.0040, but recomputing
it causally, restricted to the entity's *earlier* rows, dropped it to +0.0004.
The entire apparent gain was a July row being adjusted by a September one.

## 6. Results

| | local `tail_late` | public LB |
|---|---|---|
| raw-column baseline | 0.2259 | — |
| full pipeline | 0.5530 | **0.52708 → 0.55203** |

Across ten submissions: roughly 58% of the gain is the Stage 0/1 feature and
validation rebuild, 24% the disclosed integrity feature of §8, and 8% the blend
weighting described above.

Feature engineering is worth **2.3–2.4× over raw columns**, and it holds on the
drifted tail as much as on the easy period, so it is not buying old-regime
performance at the test period's expense. After that, hyperparameters, model
family, class weighting, recency adaptation and post-processing were each
measured and none cleared the noise floor. **The features are the result; the
fitting procedure had nothing left to give.**

## 7. Limitations we would state before being asked

- **Our local window cannot resolve differences below ~0.008 AP.** Measured
  against four leaderboard results, gains above that pass through at 29–41%;
  below it the local estimate does not merely shrink, it loses its sign. 67,196
  rows and 1,068 positives are not enough, and no choice of statistic fixes it —
  a better-centred estimator (`mean(tail_late, tail_far)`, which predicts the
  board *level* eight times more accurately) still gets a 0.0025 comparison
  backwards.
- **The feature build is not bit-reproducible.** The self-supervised graph tier
  is correctly seeded, but multi-threaded CPU reductions in torch are not
  deterministic, so four `gnn_*` columns rebuild to ~3e-4 (correlation 1.000000)
  rather than exactly — enough to move a LightGBM early stop from 230 rounds to
  758. We therefore pin round counts from the run that produced each submission
  and ship the exact matrix alongside the notebook. All model fits use
  `deterministic=True` with a fixed thread count.
- **We cannot verify the private split behaves like the public one.** Propagating
  our measured bootstrap band to the private sample size gives a ~±0.027
  public-to-private swing, which is larger than the spread across the top of the
  leaderboard.

## 8. Disclosure

**External data or models:** none. No pretrained backbones, no external
datasets. Libraries only: pandas, numpy, scipy, scikit-learn, LightGBM,
XGBoost, CatBoost, PyTorch (for the self-supervised graph embeddings, trained
from scratch on competition data alone).

**`signup_inconsistency_d`.** One feature deserves explicit mention. It compares
a row's implied signup day against the one the customer's own earlier rows
established — computed past-only, from raw non-target columns only. In real
fraud work "this account's stated age contradicts its own history" is a genuine
identity-tampering signal, and it is not target leakage. In *this* dataset it is
unusually precise: of the 1,014 labelled rows where it is non-zero (0.139% of
train), **85.8% are fraud** — 48.7× lift — covering 6.8% of all fraud, while
only 0.020% of legitimate rows show any inconsistency. That sharpness may
reflect how the fraud rows were generated. We flag it
rather than bury it: it is switchable (`REACT_SIGNUP=1`), we measured its
leaderboard value in isolation as **+0.0072** with one variable changed, and the
submitted pipeline includes it. We are happy to be scored on the variant
without it if the organisers prefer. We measured that cost directly rather than
estimating it: the identical composition, weights and pinned round counts on the
243-column matrix without the feature scores **0.54548 against 0.55150**, so it
is worth **+0.00602** on the model actually submitted.
