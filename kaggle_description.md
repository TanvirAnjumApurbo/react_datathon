# REACT 2026 Datathon by IEEE SEU SB

## A Temporal Fraud Detection Challenge — REACT 2026 | IEEE SEU Student Branch

### Overview

REACT 2026 is a national Datathon organized by the IEEE Southeast University (SEU) Student Branch. Teams are given a chronologically ordered stream of raw transaction data and must predict the probability of fraud without being given any behavioral or historical features. The challenge is to engineer time aware, leakage safe signals from customer, device, merchant, and location patterns. The top 15 teams on the private leaderboard will be invited to the onsite final at Southeast University, subject to reproducibility verification.

### Background

Real-world fraud detection is rarely a static classification problem. A transaction is not fraudulent or legitimate in isolation — it is fraudulent or legitimate relative to a customer's own history, relative to the entities involved (device, merchant, location), and relative to a moment in time. The same ৳30,000 transaction is unremarkable for one customer and a five-alarm anomaly for another. The same device is normal when it belongs to one person and suspicious when it suddenly belongs to twelve.

Fraud detection is also a moving target. Fraud rings adapt. Old signals stop working. New patterns emerge that never existed in your training data. A model that only memorizes what fraud looked like yesterday will miss what fraud looks like tomorrow.

REACT 2026 asks you to build a system that understands behavior over time — not just transactions in isolation.

### Problem

You are given a large, chronologically ordered stream of transactions from a digital payment ecosystem. Each transaction is described only by raw, unprocessed fields — no behavioral, historical, or aggregated features are provided.

Your task: for every transaction in `test.csv`, predict the probability that it is fraudulent.

You must discover and engineer the behavioral signal yourself: what is normal for this customer, this device, this merchant, this location — and how does this transaction deviate from it?

### The Challenge

The columns you are given describe what happened. They do not tell you whether it was normal. That judgment requires reconstructing, for every transaction, things like:

* **Customer behavior** — how often does this customer transact, for how much, and has that changed recently?
* **Temporal behavior** — is this an unusual hour or day for this customer? Has there been a sudden burst of activity?
* **Device behavior** — has this customer used this device before? Is this device unusually shared across many accounts?
* **Merchant behavior** — is this a merchant the customer knows? Is this merchant currently seeing unusual traffic?
* **Location behavior** — is this a location the customer normally transacts from? Did they just "teleport" from somewhere else?
* **Relationship structure** — do groups of customers, devices, and merchants form suspicious clusters that no single transaction reveals on its own?

None of this is handed to you as a column. Constructing it — correctly, and without looking into the future — is the competition.

We will not tell you exactly how fraud was generated in this dataset. Reverse-engineering the generative assumptions is not the intended path to a good score; understanding behavior, generally, is.

### Constraints — Read Carefully

This competition is explicitly about time-aware, leakage-safe modeling. The following are considered rule violations, whether accidental or deliberate:

* **Target leakage** — using a transaction's own fraud label, or any transaction's future fraud label, to construct a feature.
* **Temporal leakage** — computing a "historical" feature (e.g. "customer's average transaction amount") using transactions that occur after the transaction being scored. Every engineered feature for a transaction at time $t$ may only use information strictly before $t$.
* **Test-set leakage** — fitting any model, encoder, scaler, or target-related statistic (including target encoding) using `test.csv`. Feature computation that uses only the raw, non-target columns of `test.csv` in a strictly-past-only way is fine (e.g., a device's known transaction history can include test-period rows that occurred earlier than the row being scored); using the fraud label anywhere near `test.csv` is not, since it does not exist for you.
* **Validation leakage** — random K-fold cross-validation is not appropriate for this task, because it lets information from a customer's future transactions leak into the validation score for their past transactions. Use time-based, walk-forward, expanding-window, or purged/embargoed validation instead. Submissions that appear to rely on validation schemes inconsistent with leaderboard behavior may be asked to reproduce their pipeline.

Do not chase the public leaderboard. The public leaderboard is only 60% of the test set and is not guaranteed to represent the private 40% well. Trust a robust, time-aware local validation setup over repeated public submissions.

### Evaluation

**Official metric:** PR-AUC (Average Precision).

Fraud is rare (roughly 1.5–2% of transactions). Under this level of class imbalance, accuracy and even ROC-AUC can look deceptively strong for a model that mostly predicts "not fraud." PR-AUC evaluates the tradeoff between precision and recall specifically on the positive (fraud) class, which is what actually matters operationally — of the transactions you flag, how many are truly fraudulent, and of the fraud that exists, how much did you catch.

Formally, PR-AUC is the area under the precision–recall curve, computed as:

$$AP = \sum_{n} (R_n - R_{n-1}) \cdot P_n$$

summed over prediction thresholds $n$, ordered by decreasing recall, where $P_n$ and $R_n$ are the precision and recall at the $n$-th threshold. This is exactly `sklearn.metrics.average_precision_score`.

**Leaderboard:** 60% public / 40% private. You may select up to 2 submissions to be scored for the private leaderboard; if you select none, your best public-scoring submission is used automatically.

### Submission Format

Submit a CSV with exactly these two columns:

```csv
transaction_id,fraud
T000731942,0.0132
T000731943,0.8127
...

```

`fraud` must be a probability in $[0, 1]$, not a hard label — PR-AUC is threshold-independent and rewards well-calibrated ranking, not a binary cutoff.



# Dataset Description

### Files

| File | Description |
| --- | --- |
| `train.csv` | Historical transactions, each labeled `fraud` (0 or 1). |
| `test.csv` | Future transactions. No `fraud` column — this is what you predict. |
| `sample_submission.csv` | Example submission format. |
| `data_dictionary.csv` | Column-by-column description of the raw fields. |

Every transaction in `test.csv` occurs strictly after every transaction in `train.csv`. See `data_dictionary.csv` for the full column reference; at a glance, you receive per transaction: an id, a customer id, a timestamp, an amount (BDT), a merchant id and category, a device id and type, a coarse location, a payment method, a transaction type, and the customer's account age in days at the time of the transaction.

A small amount of missingness exists in a few non-critical categorical fields (`merchant_category`, `device_type`, `location`) — this is realistic and intentional; decide how to handle it.

### Dataset Time Range

The dataset is chronologically ordered:

* **`train.csv`**: `2026-01-01 00:00:43` to `2026-07-15 23:58:21`
* **`test.csv`**: `2026-07-16 00:00:21` to `2026-09-15 22:34:38.341114`

Every transaction in `test.csv` occurs strictly after every transaction in `train.csv`.

> **Note:** Do not assume that fraud patterns in `test.csv` are identical to those in `train.csv`. Behavioral fraud patterns can and do drift over time.

### What You Must Predict

For each `transaction_id` in `test.csv`, predict the probability that `fraud = 1`.



### Important

This dataset intentionally provides only raw transaction-level fields. No behavioral, historical, or aggregated features are provided.

##### Discovering and engineering these features — without leaking future information — is the core challenge of this competition.

See the full **Competition Overview** for temporal-leakage rules and evaluation details.


# Rules

* **Team Composition** — Teams of 2–4 students from a recognized university. Cross-university teams are allowed. One team per participant.
* **Daily Submissions** — Maximum 5 submissions per day per team.
* **External Data & Code** — No external data of any kind — no external transaction datasets, fraud labels, customer datasets, or merchant risk databases, and no external data used for fine-tuning. Publicly available open-source libraries, standard pretrained general-purpose backbones (where genuinely relevant), and public code are allowed; disclose any external model or code source you use in your final summary.
* **Code Submission & Verification** — No code or report is required to compete in the online round. Top 15 private leaderboard teams must submit a notebook and short summary for reproducibility verification to qualify for the onsite final.
* **Final Scoring Breakdown** — Final score: 60% online phase + 40% onsite presentation (3 minutes, hard stop, short Q&A).
* **Fair Play & Integrity** — Attempting to reverse-engineer or scrape the organizer's private fraud-generation logic, or targeting specific entity IDs rather than behavioral patterns, undermines the spirit of the competition and may be flagged during reproducibility review.

> **Good luck.** Think in terms of *who, when, where, how much, how often, with which device, with which merchant, and in what relationship to everyone else* — and turn that into leakage-safe features.


# Message from Organizers

Before starting, carefully read the Overview and Data tabs.

Pay particular attention to:
* Problem statement
* Dataset structure
* Evaluation metric
* Competition constraints
* Permitted approaches
* Validation requirements
* Data leakage restrictions

The dataset contains time-ordered data, so the rules regarding feature engineering and validation are especially important.

Build your fraud-detection model using only the data and resources permitted by the competition rules.

Because the dataset is time-ordered, carefully consider the rules before designing:
* Features
* Feature engineering
* Train/validation split
* Cross-validation strategy
* Model-selection process

Your approach must not introduce future information or data leakage into your features or validation process.

Please remember:
* No external datasets may be used.
* External fraud-label sources are prohibited.
* Fine-tuning on external data is prohibited.
* Do not share code, data, or predictions with teams outside your own team.
* Attempts to de-anonymize the test data or reverse-engineer how the dataset was generated are strictly prohibited and may be investigated during reproducibility review.
* Plagiarism, collusion, or misrepresentation of results may result in disqualification.
* Registration fees are non-refundable in case of disqualification.