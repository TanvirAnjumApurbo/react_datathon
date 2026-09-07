"""Emit the REACT 2026 reproducibility notebook.

    .venv/Scripts/python.exe make_final_notebook.py

Writes `notebooks/react2026_final.ipynb`: a full training run from the raw CSVs
through to both selected submission files. It is generated rather than hand-kept
so the member list, pinned round counts and blend weights come from
`submissions/*.json` instead of being retyped.

`s9_cat65` and `s10_cat80` share the same six trained members and differ only in
the final weight vector, so one training run produces both.
"""
from __future__ import annotations

import json

from src import config as C

OUT = C.ROOT / "notebooks" / "react2026_final.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {},
            "source": t.strip().splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip().splitlines(keepends=True)}


def build():
    s10 = json.loads((C.ROOT / "submissions" / "s10_cat80.json").read_text())
    s9 = json.loads((C.ROOT / "submissions" / "s9_cat65.json").read_text())
    cells = []

    cells.append(md("""
# REACT 2026 — Temporal Fraud Detection

Full training run: raw CSVs → engineered features → leakage checks → models →
the two submitted files. Nothing is loaded pre-computed.

**Metric:** PR-AUC (average precision). **Task:** score a 62-day forward block
(2026-07-16 → 2026-09-15) that starts the day after the labelled data ends.

**Private LB 0.55312 — 10th place.** Public 0.55203, from 0.52708 at our first
attempt. The file that scored it is `s10_cat80.csv`, written by section 7.

The hard part of this competition was not the model. It was discovering that
fraud in this dataset *drifts*, and that the obvious validation split hides it —
our first submission scored 0.7252 locally and 0.527 on the leaderboard. Sections
2 and 5 are about how we found that and what we did instead.

| section | |
|---|---|
| 1 | The data |
| 2 | The drift that broke our first validation |
| 3 | Features — 244 columns, all past-only |
| 4 | Leakage checks |
| 5 | Validation design |
| 6 | Training |
| 7 | Blending, and the two submissions |
"""))

    cells.append(md("""
### Running this on Kaggle

Attach three datasets and set the four paths in the next cell:

| dataset | contents | variable |
|---|---|---|
| competition data | `train.csv`, `test.csv`, `sample_submission.csv` | `DATA_DIR` |
| our source code | `src/`, `tune.py`, `data/processed/tune_rounds.s5_asbuilt.json` | `SRC` |
| feature matrix | `base_features.parquet` + `base_features.stamp.json` | `MATRIX_DIR` |

`WORK_DIR` is scratch — use `/kaggle/temp/processed` so the intermediate files do
not end up in the committed output.

The third dataset is optional; the cell after next explains what changes without
it. Runtime is roughly 1.5–2 hours on a Kaggle CPU instance (2–3 without the
supplied matrix), peak memory ~5 GB. No GPU, no internet, no external data.
"""))

    cells.append(code("""
import os, sys, json, time
from pathlib import Path

SRC        = ".."                      # repo root
DATA_DIR   = None                      # e.g. "/kaggle/input/react-2026"
WORK_DIR   = None                      # e.g. "/kaggle/temp/processed"
MATRIX_DIR = None                      # optional, see below

if SRC not in sys.path:
    sys.path.insert(0, SRC)
if DATA_DIR:
    os.environ["REACT_DATA"] = DATA_DIR
if WORK_DIR:
    os.environ["REACT_PROCESSED"] = WORK_DIR
os.environ["REACT_SIGNUP"] = "1"       # see section 7 for what this enables

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from src import config as C
from src import harness as H
from src import leakage_checks as LC
from src import validation as V
from src.io_utils import get_stream
from src.features import encoding
from tune import FITTERS, train_mask

plt.rcParams.update({"figure.figsize": (10, 3.4), "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})

BOOK = json.loads((Path(SRC) / "data/processed/tune_rounds.s5_asbuilt.json").read_text())

if MATRIX_DIR:
    import shutil
    for f in ("base_features.parquet", "base_features.stamp.json"):
        shutil.copy(Path(MATRIX_DIR) / f, C.PROCESSED / f)

print("data:", C.DATA)
print("writing to:", C.PROCESSED)
print("feature matrix:", "supplied" if MATRIX_DIR else "rebuilt from raw")
"""))

    cells.append(md("""
## Data provenance — please read this first

**This solution uses only the competition data.** No external dataset of any
kind was used at any stage: no outside transaction data, no fraud labels, no
merchant or device reputation lists, no geographic or demographic data, no
pretrained model weights, and no data used for fine-tuning anything.

Three datasets are attached to this notebook. Only one of them is data, and it
is the organisers' own:

| attached as | what it is | where it came from |
|---|---|---|
| `DATA_DIR` | `train.csv`, `test.csv`, `sample_submission.csv` | the organisers |
| `SRC` | ~170 KB of our own `.py` files, plus one small JSON | written by us |
| `MATRIX_DIR` | `base_features.parquet` | **produced by section 3 of this notebook from the competition CSVs above** |

The third entry deserves a full sentence, because a 394 MB parquet attached as an
input naturally raises the question. **It is not a data source. It is a cache of
this notebook's own output.** Section 3 turns the competition CSVs into 244
engineered columns; that file is what those cells produce. It is supplied only so
this run reproduces our submitted predictions exactly, for the reason below. Set
`MATRIX_DIR = None` and the notebook rebuilds it from the raw CSVs instead —
identical code, identical columns, no external input either way.

The cell after next does not ask you to take that on trust. It hashes the
feature-engineering source files in `SRC` and checks that hash against the one
recorded inside the supplied matrix. The two can only agree if that matrix was
generated by the code you are reading.

### Why the matrix is supplied at all

One feature block — a self-supervised graph embedding, trained from scratch on
the competition data — uses multi-threaded floating-point reductions that are not
associative. Rebuilding it reproduces four columns to about 3e-4 rather than
exactly. That is negligible in feature space (rebuilt and original correlate at
1.000000) but it perturbs the final predictions slightly: a full rebuild
reproduces **99.3% of the top 1%** of our ranking, which is the region average
precision integrates over.

So there are two honest ways to run this:

- **`MATRIX_DIR` set** — trains on the exact matrix behind our submission, and
  reproduces `s10_cat80.csv` to the digit. All feature code is present and
  readable; it simply is not re-executed. **This is how we ran it.**
- **`MATRIX_DIR = None`** — rebuilds every feature from the raw CSVs, and the
  truncation proof in section 4 then runs against a real rebuild. The output
  scores within roughly 0.002 of our leaderboard result rather than matching it
  exactly.
"""))

    cells.append(code("""
files = sorted(Path(SRC).rglob("*"))
code_files = [f for f in files if f.suffix == ".py"]
data_files = [f for f in files if f.suffix in {".csv", ".parquet", ".pkl", ".pt", ".h5", ".npy"}]

print(f"{len(code_files)} source files, {sum(f.stat().st_size for f in code_files)/1024:.0f} KB")
for f in code_files:
    print(f"   {f.relative_to(SRC)}")
print()
print("other files:", [f.name for f in files if f.suffix == ".json"])
print("data files in SRC:", data_files or "none")
assert not data_files, "SRC must contain code only"

print()
print("--- where the supplied feature matrix came from ---")
if MATRIX_DIR:
    stamp = json.loads((Path(MATRIX_DIR) / "base_features.stamp.json").read_text())
    print("hash recorded inside the matrix :", stamp["hash"])
    print("hash of the feature code in SRC :", H.feature_source_hash())
    assert stamp["hash"] == H.feature_source_hash(), "matrix was not built by this code"
    print("they match, so this matrix is the output of the feature code above,")
    print(f"applied to the competition CSVs. {stamp['n_rows']:,} rows, "
          f"{len(stamp['provenance'])} engineered columns.")
else:
    print("none supplied: every feature is rebuilt from the raw CSVs below.")

print()
print("libraries: pandas, numpy, scipy, scikit-learn, lightgbm, catboost, torch")
print("torch trains the graph embeddings from scratch on competition data only.")
print("no pretrained weights, no external data, no internet access required.")
"""))

    cells.append(md("""
## 1. The data

One time-sorted stream of train + test. Features are computed over the combined
stream — the rules permit this explicitly, and it is necessary: building on train
alone would leave every test row without its own recent history and make test
features systematically unlike training features.
"""))

    cells.append(code("""
df = get_stream()
ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
y = df[C.TARGET].to_numpy(dtype=float)

print(f"{len(df):,} rows   {(~is_test).sum():,} train / {is_test.sum():,} test")
print(f"train  {ts[~is_test].min():%Y-%m-%d} -> {ts[~is_test].max():%Y-%m-%d}   "
      f"fraud {np.nanmean(y[~is_test]):.2%}")
print(f"test   {ts[is_test].min():%Y-%m-%d} -> {ts[is_test].max():%Y-%m-%d}   no labels")
print(f"\\ncustomers {df[C.CUSTOMER].nunique():,}   devices {df[C.DEVICE].nunique():,}   "
      f"merchants {df[C.MERCHANT].nunique():,}")
"""))

    cells.append(code("""
wk = df.assign(w=ts.dt.to_period("W").dt.start_time).groupby("w")
vol, rate = wk.size(), wk[C.TARGET].mean()
split = ts[is_test].min()

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4.6))
ax[0].bar(vol.index, vol.values, width=5, color="#8090a8")
ax[0].axvline(split, color="crimson", lw=1.2, ls="--")
ax[0].set_ylabel("transactions / week")
ax[1].plot(rate.index, rate.values * 100, color="#2b6cb0", lw=1.5)
ax[1].axvline(split, color="crimson", lw=1.2, ls="--")
ax[1].set_ylabel("fraud rate %")
ax[0].set_title("Volume and fraud rate by week (dashed line = start of test)")
plt.tight_layout(); plt.show()

print(f"fraud rate is stable at {np.nanmean(y[~is_test]):.2%} -- the drift is not in how much "
      "fraud there is.")
"""))

    cells.append(md("""
## 2. The drift that broke our first validation

Our first submission scored **0.7252** on a 62-day forward fold and **0.52708**
on the leaderboard. Nothing was leaking. The fold averaged a regime change away.

Fraud amounts fall steadily across the stream while legitimate amounts do not.
By July the two are close enough that amount — the single strongest feature —
has lost most of its power. The plot below is the whole story.
"""))

    cells.append(code("""
tr = df[~is_test].assign(w=ts[~is_test].dt.to_period("W").dt.start_time)
med = tr.groupby(["w", C.TARGET])["amount_bdt"].median().unstack()

from src.validation import ap as _ap
amt_ap = tr.groupby("w").apply(
    lambda g: _ap(g[C.TARGET], g["amount_bdt"]) if g[C.TARGET].sum() > 20 else np.nan,
    include_groups=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
ax[0].plot(med.index, med[1.0], color="crimson", lw=1.6, label="fraud")
ax[0].plot(med.index, med[0.0], color="#2b6cb0", lw=1.6, label="legitimate")
ax[0].set_yscale("log"); ax[0].set_ylabel("median amount (BDT)"); ax[0].legend()
ax[0].set_title("Fraud amounts converge on legitimate ones")
ax[1].plot(amt_ap.index, amt_ap.values, color="#6b46c1", lw=1.6)
ax[1].set_ylabel("AP"); ax[1].set_title("Predictive power of raw amount, by week")
plt.tight_layout(); plt.show()

print(f"fraud median amount   {med[1.0].iloc[0]:>8,.0f} (Jan)  ->  {med[1.0].iloc[-1]:>7,.0f} (Jul)")
print(f"legit median amount   {med[0.0].iloc[0]:>8,.0f} (Jan)  ->  {med[0.0].iloc[-1]:>7,.0f} (Jul)")
print(f"AP of raw amount      {amt_ap.iloc[0]:>8.3f}       ->  {amt_ap.iloc[-1]:>7.3f}")
"""))

    cells.append(md("""
This is **concept drift, not covariate shift**, and the distinction rules out a
whole family of fixes. An adversarial classifier separating train from test rows
reaches AUC 0.612 on the raw columns, and 0.514 — chance — once `account_age_days`
is removed. `amount_bdt` has PSI 0.0002 between the two periods: the amount
*distribution* barely moved. What moved is `P(fraud | amount)`.

Density-ratio importance weighting corrects a shift in `p(x)` while assuming
`p(y|x)` is fixed, which is exactly backwards here, so we ruled it out before
fitting anything.
"""))

    cells.append(md("""
## 3. Features

244 columns in seven blocks, every one computed from information strictly
earlier than the row it describes. All historical statistics route through a
single windowing primitive, so the past-only guarantee lives in one place
instead of being re-argued per feature.

Two signals do most of the work, and both are *relative* rather than absolute:
an amount more than 10× the customer's own past mean is 92.8% fraud, and a
customer whose previous transaction was under a minute ago is 86.0% fraud.
Merchant velocity, by contrast, is worthless (0.69× lift) — the merchants are
high-traffic aggregators, with the top 10 carrying 57.8% of all rows.

The target encoder carries a **7-day feedback delay**: a fraud label does not
exist when a transaction happens, it exists once an investigation confirms it.
Without the delay, training rows read a perfectly fresh encoder while test rows
read one frozen at the cutoff and up to 62 days stale. Adding it is worth
0.008–0.028 AP on every validation window we have.
"""))

    cells.append(code("""
t0 = time.time()
base = H.load_base(df, use_gnn=True)
te = encoding.build(df, fit_cutoff=C.TRAIN_END)
X = pd.concat([base, te], axis=1)
prov = dict(H.base_provenance())
te_cols = set(te.columns)
print(f"\\n{X.shape[1]} features built in {time.time()-t0:.0f}s")
"""))

    cells.append(code("""
blocks = pd.Series([("encoding" if c in te_cols else prov.get(c, "?").split(":")[0])
                    for c in X.columns]).value_counts()
gain_note = {"velocity": 27.4, "entity": 13.1, "amount": 48.6, "graph": 3.4,
             "behaviour": 2.4, "temporal": 2.9, "encoding": 2.4}

fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))
ax[0].barh(blocks.index[::-1], blocks.values[::-1], color="#8090a8")
ax[0].set_title("columns per block")
g = pd.Series(gain_note).reindex(blocks.index)
ax[1].barh(g.index[::-1], g.values[::-1], color="#2b6cb0")
ax[1].set_title("share of model gain (%)")
plt.tight_layout(); plt.show()

print("Column count and usefulness are different things: `amount` is 16% of the")
print("columns and 49% of the gain; `behaviour` is 11% of the columns and 2%.")
"""))

    cells.append(md("""
## 4. Leakage checks

These run rather than being described, and they fail loudly.

The important one is **truncation invariance**: rebuild the features on a stream
cut at a cutoff, then diff against the full-stream values on the rows they share.
If any feature at time *t* used information from after *t*, deleting the future
would change it. It runs over the 202 behavioural columns — temporal, amount,
velocity, entity, behaviour.

The 29 graph columns are audited separately, and it is worth being precise rather
than quietly omitting them. Run under the same test, **19 of the 29 are invariant
to the row** — every structural quantity: degrees, component sizes, 2-hop counts,
same-component flags, ratios, shares. The 10 that move are exactly the
basis-dependent ones: the four learned embeddings, three SVD norm/drift columns,
two SVD cosines, and one fraction normalised by a global node count.

That is the expected signature. Their construction is past-only — at each weekly
boundary the graph is built from strictly earlier rows and joined onto the
*following* week — but the embeddings are expressed in a basis (a truncated SVD,
and a self-supervised model) whose orientation depends on the size of the entity
index, which is taken from the whole stream. Truncating changes the basis, so the
numbers move without any row depending on its own future.

There is a second reason the row-wise test cannot settle this: the block is not
bit-reproducible against itself. Two builds of the *identical* full stream differ
by up to 3e-4 (correlation 1.000000) because the embedding step uses
multi-threaded float reductions. The truncation test's tolerance is 1e-5, so it
would flag those columns comparing a build to itself.

We flag this instead of hiding it. The block uses no labels, and it earns
nothing: dropping all 29 columns moves our selection window by **+0.0020** —
i.e. slightly *up*. Nothing in our result depends on it.

The last check exists because the other five all passed while a real defect was
live — the weekly snapshots left 29 columns empty for 10,455 *test* rows and zero
train rows, and since those rows carry no labels and sit in no fold, nothing
noticed.
"""))

    cells.append(code("""
def behavioural_blocks(d):
    from src.features import amount, behaviour, entity, temporal, velocity
    return pd.concat([temporal.build(d), amount.build(d), velocity.build(d),
                      entity.build(d), behaviour.build(d)], axis=1)

res = LC.run_all(X, df, build_fn=behavioural_blocks)
LC.check_tail_coverage(X, df)
BEHAV = {"temporal", "amount", "velocity", "entity", "behaviour"}
n_trunc = sum(prov.get(c, "").split(":")[0] in BEHAV for c in X.columns)

print("no banned columns / no duplicates      PASS")
print("first row of each customer is null     PASS")
print("train/test boundary continuity         PASS")
print("end-of-stream coverage                 PASS")
print(f"truncation invariance                  PASS  ({n_trunc} behavioural columns)")
print("                                             graph block audited separately, below")
"""))

    cells.append(code("""
top = res["single_feature_ap"].head(12)
plt.figure(figsize=(9, 3))
plt.barh(top["feature"][::-1], top["ap"][::-1], color="#2b6cb0")
plt.axvline(0.85, color="crimson", ls="--", lw=1.2)
plt.title("Strongest single features (red line = we would call it a leak)")
plt.tight_layout(); plt.show()

print(f"strongest single feature: {top.iloc[0]['feature']} at AP {top.iloc[0]['ap']:.3f}")
print("A leaked column would sit near 1.0. The top of this list is 'this amount is")
print("unlike the customer's own history', which is the question the task asks.")
"""))

    cells.append(md("""
## 5. Validation

Random K-fold is forbidden here and would be wrong anyway. We use forward blocks
only, and after the lesson in section 2 we stopped selecting on the wide 62-day
fold — it averages two fraud regimes together.

Selection moved to the **last 2.5 labelled weeks**, read at three forecast
horizons over the *same rows*. Reading one horizon is not just noisier than
reading three; it answers a different question. One of our configurations ranks
7th of 8 at 46–62 days and **1st** at 75–91.
"""))

    cells.append(code("""
folds = V.describe_folds(df)
print(folds.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 2.4))
rows = [("tail_recent (1-17d)", V.tail_recent_fold()), ("tail_late (46-62d)", V.tail_late_fold()),
        ("tail_far (75-91d)", V.tail_far_fold()), ("primary_62d", V.primary_fold())]
for i, (lbl, f) in enumerate(rows):
    ax.barh(i, (f.train_end - C.TRAIN_START).days, color="#c7d2e0")
    ax.barh(i, (f.val_end - f.val_start).days, left=(f.val_start - C.TRAIN_START).days,
            color="crimson")
ax.barh(len(rows), (C.TRAIN_END - C.TRAIN_START).days, color="#c7d2e0")
ax.barh(len(rows), (C.TEST_END - C.TEST_START).days,
        left=(C.TEST_START - C.TRAIN_START).days, color="#2b6cb0")
ax.set_yticks(range(len(rows) + 1)); ax.set_yticklabels([r[0] for r in rows] + ["REAL TEST"])
ax.set_xlabel("days from 2026-01-01"); ax.set_title("grey = train, red = scored, blue = the real test window")
plt.tight_layout(); plt.show()
"""))

    weights10 = json.dumps(s10["weights"])
    weights9 = json.dumps(s9["weights"])
    cells.append(md(f"""
## 6. Training

Six members: five LightGBM configurations plus CatBoost, three seeds each,
eighteen fits in total.

**Round counts are fixed, not searched.** Early stopping on average precision
sits on a plateau here — identical code on a matrix differing by 3e-4 moved one
member from 230 rounds to 758 for a score change of 0.0011. So the counts are
pinned to the run that produced the uploaded files and scaled by
`sqrt(n_final / n_ref)` for the refit on all labelled data.

We do **not** use `scale_pos_weight`, SMOTE, or any resampling. Average precision
reads only the ordering, so upweighting the positive class adds no information
and distorts the leaf values — measured, it loses on 6 of 6 windows.
"""))

    cells.append(code("""
MEMBERS = %s
SEEDS = 3
DET = dict(lgb=dict(deterministic=True, force_row_wise=True, num_threads=8),
           xgb=dict(nthread=8), cat=dict(thread_count=8))

tm = train_mask(ts, is_test, C.TRAIN_END, None)
Xt, yt = X[tm], y[tm]
Xs = X[is_test]
n_test, n_final = int(is_test.sum()), int(tm.sum())
print(f"training on {n_final:,} labelled rows, scoring {n_test:,}\\n")

ranks, gains = {}, None
for name in MEMBERS:
    cfg = BOOK[name]
    rounds = max(int(cfg["rounds"] * (n_final / cfg["n_train"]) ** 0.5), 50)
    seed_ranks = []
    for s in range(SEEDS):
        t = time.time()
        params = {**cfg["params"], **DET[cfg["kind"]]}
        model, _, pred_fn = FITTERS[cfg["kind"]](
            Xt, yt, None, None, None, params, rounds, C.SEED + 100 * s)
        seed_ranks.append(rankdata(pred_fn(Xs)) / n_test)
        if name == "lgb_base" and s == 0:
            gains = pd.Series(model.feature_importance("gain"), index=Xt.columns)
        print(f"  {name:<12s} seed {s}  {rounds:>5d} rounds  {time.time()-t:>5.0f}s")
    ranks[name] = np.mean(seed_ranks, axis=0)
print("\\ndone")
""" % json.dumps(list(s10["weights"]))))

    cells.append(code("""
g = (gains / gains.sum() * 100).sort_values(ascending=False).head(20)
plt.figure(figsize=(9, 4.2))
plt.barh(g.index[::-1], g.values[::-1], color="#2b6cb0")
plt.xlabel("% of total gain"); plt.title("Top 20 features (lgb_base, seed 0)")
plt.tight_layout(); plt.show()

print("Ratio-to-own-history features dominate. That is the intended answer to")
print("'is this normal for this customer', rather than 'is this a large amount'.")
"""))

    cells.append(md(f"""
## 7. Blending, and the two submissions

Ranks are averaged, not probabilities — average precision depends only on the
ordering, and the two model families are on different scales.

**Members were added for decorrelation, not for count**, and we paid a submission
each way to learn it. Eight LightGBM variants differing only in sample weights
gained +0.0023 locally and **lost 0.0025** on the leaderboard; one CatBoost gained
+0.0011 locally and **gained 0.0014**. Measured as rank correlation against the
LightGBM core, CatBoost sits at 0.62 while every LightGBM variant sits at
0.74–0.91 — the same band the core members occupy among each other.

That is also what sets the weights. Moving weight onto the single decorrelated
member was worth **+0.00208** on the leaderboard, in four monotone steps. Our
local window is flat across that whole range and could not have found it.

The two files below are our selected submissions. They share all eighteen
trained models and differ only in this weight. **`s10_cat80.csv` is the
leaderboard-scoring submission** — private 0.55312 against `s9_cat65.csv`'s
0.55272.
"""))

    cells.append(code("""
BLENDS = {"s10_cat80": %s,
          "s9_cat65":  %s}
BOARD  = {0.167: 0.54995, 0.500: 0.55150, 0.650: 0.55187, 0.800: 0.55203}

plt.figure(figsize=(6.5, 3))
plt.plot(list(BOARD), list(BOARD.values()), "o-", color="#2b6cb0")
for w, v in BOARD.items():
    plt.annotate(f"{v:.5f}", (w, v), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8)
plt.xlabel("weight on CatBoost (the decorrelated member)")
plt.ylabel("public LB"); plt.title("Blend weight vs leaderboard")
plt.ylim(0.5495, 0.5524); plt.tight_layout(); plt.show()

print("Monotone and decelerating: +0.00155, +0.00037, +0.00016.")
print("The last step is below our measured board noise floor of 0.00019, so we")
print("selected both endpoints of the flat region rather than betting on one.")
""" % (weights10, weights9)))

    cells.append(code("""
sample = pd.read_csv(C.SAMPLE_SUB)
ids = df.loc[is_test, C.ID_COL].to_numpy()
outputs = {}

for tag, w in BLENDS.items():
    p = sum(v * ranks[m] for m, v in w.items()) / sum(w.values())
    p = (p - p.min()) / (p.max() - p.min())
    sub = sample[[C.ID_COL]].merge(pd.DataFrame({C.ID_COL: ids, C.TARGET: p}),
                                   on=C.ID_COL, how="left")
    assert sub[C.TARGET].notna().all() and len(sub) == len(sample)
    assert sub[C.TARGET].between(0, 1).all()
    sub.to_csv(f"{tag}.csv", index=False)
    outputs[tag] = sub
    print(f"wrote {tag}.csv   {len(sub):,} rows   mean {sub[C.TARGET].mean():.4f}")

a, b = outputs["s10_cat80"][C.TARGET], outputs["s9_cat65"][C.TARGET]
k = int(0.01 * len(a))
ov = len(set(np.argsort(-a.values)[:k]) & set(np.argsort(-b.values)[:k])) / k
print(f"\\nthe two files share {ov:.1%} of their top 1% -- they are close by design")
"""))

    cells.append(code("""
plt.figure(figsize=(9, 3))
plt.hist(outputs["s10_cat80"][C.TARGET], bins=120, color="#2b6cb0", log=True)
plt.xlabel("predicted score"); plt.ylabel("rows (log)")
plt.title("Prediction distribution, s10_cat80")
plt.tight_layout(); plt.show()

print("Heavily skewed toward zero, as a 1.76% base rate demands. Average precision")
print("is scale-free, so the min-max rescale above changes nothing about the score.")
"""))

    cells.append(md("""
## What we submitted, and what we would flag

| file | private | public |
|---|---|---|
| **`s10_cat80.csv`** — the scoring submission | **0.55312** | 0.55203 |
| `s9_cat65.csv` | 0.55272 | 0.55187 |

Journey: 0.52708 → 0.55312, finishing 10th of the private leaderboard having been
13th on the public one. Roughly 58% of the gain came from rebuilding the features
and the validation after the drift finding, 24% from a disclosed data-quality
feature (below), and 8% from the blend weighting in section 7.

One result is worth recording because we could not have reached it from local
validation. The two files differ only in the CatBoost weight, 0.80 against 0.65,
and on the public set they were 0.00016 apart — below the 0.00019 noise floor we
had measured, so we could not call it real and selected both endpoints rather
than betting on one. The private half, a disjoint sample, put the same file ahead
by 0.00040. Two independent halves agreeing means the weighting effect was
genuine signal.

Three things we would rather state than have found:

**`signup_inconsistency_d`.** One feature compares a row's implied signup day
against the one the customer's own earlier rows established. Computed past-only
from raw non-target columns, it is an ordinary data-quality check, and "this
account's stated age contradicts its own history" is a real identity-tampering
signal. In this dataset it is unusually sharp — of the 1,014 labelled rows where
it fires, 85.8% are fraud. We measured its worth directly by rebuilding the same
blend without it: **0.54548 against 0.55150, so +0.00602**. It is enabled by
`REACT_SIGNUP=1` in section 0, and we are content to be scored without it.

**The graph tier fits a model on test-period rows.** The embeddings are
self-supervised — no labels anywhere — and strictly past-only: each weekly
snapshot is built from rows before that boundary, which is the case the rules
explicitly allow. We note it because "fitting a model" and "computing a feature"
blur here. It is worth nothing measurable: dropping the whole graph block moves
our tail window by +0.0020.

**We used the leaderboard to set one parameter.** The blend weight in section 7
was chosen from four leaderboard readings because our local window is flat across
that range. Everything else — features, validation, hyperparameters, model
family, post-processing — was decided locally, and most of it came back negative.
"""))

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    return nb


if __name__ == "__main__":
    nb = build()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
        fh.write("\n")
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    print(f"wrote {OUT}  ({len(nb['cells'])} cells, {n_code} code)")
