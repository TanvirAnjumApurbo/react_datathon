"""Stage 2: model search, judged on the window that tracks the leaderboard.

    python -u tune.py --set params     # LightGBM hyperparameters
    python -u tune.py --set drift      # recency weighting + training-window truncation
    python -u tune.py --set family     # XGBoost / CatBoost, re-measured on the tail
    python -u tune.py --set all

Why this exists next to `ablate.py`
-----------------------------------
`ablate.py` answers "does this *feature set* help", and routes everything
through `harness.run_folds`, which deliberately exposes no knobs beyond the
matrix. Stage 2 asks a different question -- "does this *fitting procedure*
help" -- and two of its candidates cannot be expressed as a feature set:

* **recency weighting** needs a weight vector computed relative to each fold's
  own cutoff, not one fixed array reused across cutoffs;
* **window truncation** needs to change the training mask itself, which
  `fit_cutoff` derives from the fold.

So the fitter lives here. It keeps harness's two honesty conventions verbatim:
early-stop on the widest window at a cutoff and report the narrow ones as free
reads; where a cutoff has only the window being scored, borrow a round count
rather than early-stop on the thing being judged.

What is being selected on
-------------------------
`tail_late` (46-62 days ahead, newest regime) is the selection target and
`tail_recent` (1-17 days ahead, same rows) is the confirmation. The real test
window spans 1-62 days past the cutoff, so it sits *between* the two
geometries; a candidate that wins on one and loses on the other has not earned
a submission. `primary_62d` is reported for continuity with the older numbers
and is not a selection target -- it averages two fraud regimes together, which
is how submission 1 came to expect 0.725 and get 0.527.

Every candidate's tail predictions are persisted so `blend.py` can pick weights
from evidence rather than assuming an equal-weight average is safe. It is not:
on the previous feature set both non-LightGBM families scored below every
LightGBM config, and rank-averaging all of them landed below the best single.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src import evaluate as E
from src import harness as H
from src import validation as V
from src.io_utils import get_stream
from src.features import encoding

RESULTS = C.PROCESSED / "tune_results.csv"
TAIL_PREDS = C.PROCESSED / "tune_tail_preds.parquet"
ROUND_BOOK = C.PROCESSED / "tune_rounds.json"

# Two runs of identical code differ by about this much (measured: top-500
# overlap 98.4%). A delta under it is not a result.
NOISE_FLOOR = 0.003

LGB_BASE = dict(
    objective="binary", metric="average_precision",
    num_threads=0, verbosity=-1, seed=C.SEED,
)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
class Cand:
    """One fitting procedure: model family, params, and the drift knobs.

    `halflife` weights a training row by 0.5 ** (age_days / halflife).
    `train_days` drops training rows older than that outright. They are
    different bets -- weighting keeps the old rows' structure at reduced
    influence, truncation denies the model the old regime entirely -- and
    under concept drift the literature does not agree on which wins, so both
    are measured.

    `drop_blocks` hides whole feature blocks from this member. That is
    `ablate.py`'s question, not Stage 2's, with one difference that matters: an
    ablation asks whether the *pipeline* should carry a block, and the answer
    for `amount` is a clear yes (-0.0069 on the tail). This asks whether a
    **blend member** that cannot see it makes different mistakes than one that
    can. Submission 3 vs 5 is the only board-measured lesson available -- eight
    LightGBM variants differing in sample weights cost 0.0025, one CatBoost
    gained 0.0014 -- so a member has to earn its place by decorrelation, and
    denying it the most drift-exposed block is a mechanism for producing some.

    `seed_offset` shifts every seed this member is fitted with, so two members
    that share a fitting procedure but should explore different random subspaces
    do not collapse onto the same fit.
    """

    def __init__(self, name, kind="lgb", params=None, halflife=None, train_days=None,
                 drop_blocks=None, seed_offset=0):
        self.name = name
        self.kind = kind
        self.params = params or {}
        self.halflife = halflife
        self.train_days = train_days
        self.drop_blocks = set(drop_blocks) if drop_blocks else None
        self.seed_offset = seed_offset


# The report's LightGBM ranges: lr 0.01-0.05, num_leaves 31-255,
# min_child_samples 50-500, feature_fraction 0.5-0.9, bagging 0.6-0.9,
# lambda_l1/l2 0-10. `lgb_base` is harness's default, so it is the reference
# every other row is a delta against.
P_BASE = dict(learning_rate=0.05, num_leaves=64, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0)

PARAM_CANDS = [
    Cand("lgb_base", params=P_BASE),
    Cand("lgb_deep", params=dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=50,
                                 feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                                 lambda_l2=10.0)),
    Cand("lgb_shallow", params=dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                                    feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1,
                                    lambda_l2=1.0)),
    # The handbook's warning, after Random Forest's Average Precision collapsed
    # on their real data while its validation score held, was to regularize hard.
    Cand("lgb_reg", params=dict(learning_rate=0.04, num_leaves=48, min_data_in_leaf=300,
                                feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
                                lambda_l1=5.0, lambda_l2=10.0)),
    # Extremely randomized splits: weaker alone, decorrelated from the rest,
    # which is what a blend actually wants from a fifth member.
    Cand("lgb_extra", params=dict(learning_rate=0.05, num_leaves=64, min_data_in_leaf=100,
                                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                                  lambda_l2=5.0, extra_trees=True)),
]

# Half-lives short enough to matter. The previous verdict ("recency weighting
# does not pay") tested 45 d and 90 d against `primary_62d`, whose training
# data ends six weeks before the new regime -- the one fold that structurally
# cannot see adaptation working. 7-21 d is where the fraud amount signature
# actually moves.
DRIFT_CANDS = [
    Cand("lgb_base", params=P_BASE),
    Cand("hl_7d", params=P_BASE, halflife=7.0),
    Cand("hl_14d", params=P_BASE, halflife=14.0),
    Cand("hl_21d", params=P_BASE, halflife=21.0),
    Cand("hl_45d", params=P_BASE, halflife=45.0),
    Cand("hl_90d", params=P_BASE, halflife=90.0),
    Cand("win_45d", params=P_BASE, train_days=45),
    Cand("win_90d", params=P_BASE, train_days=90),
    Cand("win_120d", params=P_BASE, train_days=120),
]

FAMILY_CANDS = [
    Cand("lgb_base", params=P_BASE),
    Cand("xgb_base", kind="xgb",
         params=dict(learning_rate=0.05, max_depth=8, min_child_weight=5,
                     subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0)),
    Cand("xgb_shallow", kind="xgb",
         params=dict(learning_rate=0.03, max_depth=5, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0)),
    Cand("cat_base", kind="cat",
         params=dict(learning_rate=0.05, depth=8, l2_leaf_reg=5.0)),
]

# ---------------------------------------------------------------------------
# Members proposed for *decorrelation*, each with its mechanism stated before
# measurement -- because the alternative is picking the argmax of `tail_late`,
# which is how `significant_blocks_only` and submission 3 both happened.
#
# The board's own evidence for what qualifies: `cat_base` sits at rank rho 0.644
# against the five-config LightGBM core and gained +0.0014; the eight recency and
# window variants sit at 0.736-0.826, inside the 0.833-0.910 band the core
# members occupy among themselves, and cost 0.0025. Family labels are not the
# diversity that paid -- `xgb_base` is at 0.827, in the clone band, which is why
# it is not proposed here despite being a different library.
#
#   lgb_durable   cannot see the `amount` block at all. Amount-family features
#                 retain 0.61-0.64 of their standalone power across the regime
#                 split while `cust_dt` retains 0.89 and `gnn_cd_cos` 0.99, and
#                 `amount` carries 48.6% of model gain. A member denied it must
#                 rank by the signals that survive. Read its late->far decay,
#                 not its point estimate.
#   cat_durable   the same feature view through a different algorithm. If the
#                 two mechanisms that have each produced decorrelation compose,
#                 this is the least correlated member the project can build.
#   lgb_linear    `linear_tree=True`. 44 features are unbounded counters whose
#                 test values sit past every split point the trees ever saw
#                 (`merch_age_days` train mean 98.7 -> test 226.2), so constant
#                 leaves treat them as constant at scoring time. Linear leaves
#                 extrapolate instead of flat-lining. The literature calls this
#                 real but not a silver bullet; hold it loosely.
#   lgb_sub30     `feature_fraction=0.30`. Random-subspace / feature bagging is
#                 the standard robustness device for a redundant feature space,
#                 and this one is measurably redundant -- `amt_ratio_mean_30d`
#                 scores AP 0.183 standalone and contributes nothing marginal.
#                 Forcing a member onto different redundant substitutes is
#                 decorrelation by construction, and it is the cheapest fit here.
#
# Ordered cheapest-and-surest first. `tune.py` writes its results once, at the
# end of `main`, so a run that has to be killed loses everything in it -- and
# the last two candidates are the ones most likely to make that necessary
# (CatBoost took 490s per cutoff on the family set, and linear trees are both
# slower per round and the only candidate here that might not fit at all).
DIVERSE_CANDS = [
    Cand("lgb_base", params=P_BASE),
    Cand("lgb_durable", params=P_BASE, drop_blocks={"amount"}),
    Cand("lgb_sub30", params={**P_BASE, "feature_fraction": 0.30}, seed_offset=7),
    Cand("lgb_linear", params={**P_BASE, "linear_tree": True, "linear_lambda": 1.0,
                               "num_leaves": 31, "min_data_in_leaf": 200}),
    Cand("cat_durable", kind="cat", drop_blocks={"amount"},
         params=dict(learning_rate=0.05, depth=8, l2_leaf_reg=5.0)),
]

# A second and third CatBoost, proposed after the board answered the weighting
# question. Shifting blend weight from the LightGBM core toward `cat_base`
# (1/6 -> 1/2) gained +0.00155 on the leaderboard, against a same-composition
# board floor measured at 0.00019 -- so the decorrelated family is worth more
# weight than member-equal gives it. The local weight curve then goes flat from
# 0.4 to 0.75, which caps how much *more* weight helps.
#
# These exist to test the other axis: hold the CatBoost family at ~1/2 and split
# it across configurations that are not the same model. That cuts the
# decorrelated half's own variance without shrinking it, which the weight curve
# cannot do. `rsm` is CatBoost's random-subspace parameter and `depth=10` moves
# the capacity a long way from `cat_base`; both are CPU-only paths, smoke-tested
# before being run at scale.
CAT2_CANDS = [
    Cand("cat_base", kind="cat",
         params=dict(learning_rate=0.05, depth=8, l2_leaf_reg=5.0)),
    Cand("cat_deep", kind="cat",
         params=dict(learning_rate=0.04, depth=10, l2_leaf_reg=10.0)),
    Cand("cat_rsm", kind="cat",
         params=dict(learning_rate=0.05, depth=8, l2_leaf_reg=5.0, rsm=0.6,
                     bootstrap_type="Bernoulli", subsample=0.8)),
]

SETS = {"params": PARAM_CANDS, "drift": DRIFT_CANDS, "family": FAMILY_CANDS,
        "diverse": DIVERSE_CANDS, "cat2": CAT2_CANDS}


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def recency_weights(ts: pd.Series, tm: np.ndarray, cutoff, halflife: float | None):
    if halflife is None:
        return None
    age_d = (pd.Timestamp(cutoff) - ts[tm]).dt.total_seconds().to_numpy() / 86_400.0
    return np.power(0.5, age_d / halflife)


def train_mask(ts: pd.Series, is_test: np.ndarray, cutoff, train_days: int | None):
    tm = (~is_test) & (ts <= cutoff).to_numpy()
    if train_days is not None:
        tm &= (ts > pd.Timestamp(cutoff) - pd.Timedelta(days=train_days)).to_numpy()
    return tm


def _fit_lgb(Xt, yt, w, Xe, ye, params, rounds, seed):
    import lightgbm as lgb

    p = {**LGB_BASE, **params, "seed": seed,
         "bagging_seed": seed, "feature_fraction_seed": seed}
    dtr = lgb.Dataset(Xt, label=yt, weight=w)
    if Xe is None:
        m = lgb.train(p, dtr, num_boost_round=rounds)
        return m, rounds, lambda X: m.predict(X)
    dva = lgb.Dataset(Xe, label=ye, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    n = int(m.best_iteration or m.num_trees())
    return m, n, lambda X: m.predict(X, num_iteration=m.best_iteration)


def _fit_xgb(Xt, yt, w, Xe, ye, params, rounds, seed):
    import xgboost as xgb

    # Built then updated rather than `dict(nthread=0, **params)`: the caller is
    # allowed to pin `nthread` for a deterministic fit, and the literal form
    # raises TypeError on the duplicate keyword instead of letting it win.
    p = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
             nthread=0, seed=seed)
    p.update(params)
    dtr = xgb.DMatrix(Xt, label=yt, weight=w, enable_categorical=True)
    if Xe is None:
        m = xgb.train(p, dtr, num_boost_round=rounds)
        return m, rounds, lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True))
    dva = xgb.DMatrix(Xe, label=ye, enable_categorical=True)
    m = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "v")],
                  early_stopping_rounds=150, verbose_eval=False)
    it = getattr(m, "best_iteration", None)
    n = int(it + 1) if it is not None else rounds
    return m, n, (lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True),
                                      iteration_range=(0, n)))


def _fit_cat(Xt, yt, w, Xe, ye, params, rounds, seed):
    from catboost import CatBoostClassifier, Pool

    # The matrix is entirely numeric -- the low-cardinality categoricals are
    # already integer-coded upstream -- so there is nothing to declare here.
    m = CatBoostClassifier(iterations=rounds, eval_metric="PRAUC", random_seed=seed,
                           verbose=0, allow_writing_files=False, **params)
    tr = Pool(Xt, yt, weight=w)
    if Xe is None:
        m.fit(tr)
        n = rounds
    else:
        m.fit(tr, eval_set=Pool(Xe, ye), early_stopping_rounds=150)
        n = int(m.get_best_iteration() or m.tree_count_)
    return m, n, lambda X: m.predict_proba(X)[:, 1]


FITTERS = {"lgb": _fit_lgb, "xgb": _fit_xgb, "cat": _fit_cat}


def keep_columns(cols, prov: dict[str, str], drop_blocks) -> list[str]:
    """Columns this member may see, given the blocks it is denied.

    A column with no provenance entry is kept. The target-encoded columns are
    rebuilt per cutoff and so never appear in the cached stamp, and silently
    dropping the encoder because it was not in a dictionary would be a much
    worse failure than carrying one unexpected column.
    """
    if not drop_blocks:
        return list(cols)
    return [c for c in cols if prov.get(c, "").split(":")[0] not in drop_blocks]


def fit_at(X, y, ts, is_test, cutoff, folds, cand, fixed_rounds=None,
           max_rounds=2000, seed=C.SEED, keep=None):
    """Fit one candidate at one cutoff; return full-stream predictions.

    Early stopping, when it happens, uses a window *disjoint* from the ones
    being reported (`validation.stopping_fold`). Not the widest one: `tail_late`
    is a subset of `primary_62d`, so stopping on the latter would choose the
    round count from the former's own labels. Where the only window is the one
    being scored, the caller supplies `fixed_rounds` instead -- `tail_recent`
    has no companion to hide behind, because the stream ends inside it.

    `keep` restricts the columns, and is applied in the same `.loc` that takes
    the row mask so a member with a reduced feature view costs one copy of the
    matrix rather than two.
    """
    tm = train_mask(ts, is_test, cutoff, cand.train_days)
    vm = np.zeros(len(X), dtype=bool)
    for f in folds:
        vm |= f.val_mask(ts, is_test)

    keep = list(X.columns) if keep is None else keep
    w = recency_weights(ts, tm, cutoff, cand.halflife)
    Xt, yt = X.loc[tm, keep], y[tm]

    if fixed_rounds is None:
        es = V.stopping_fold(folds)
        em = es.val_mask(ts, is_test)
        Xe, ye = X.loc[em, keep], y[em]
        rounds = max_rounds
    else:
        Xe, ye, rounds = None, None, fixed_rounds

    t0 = time.time()
    _, n_rounds, pred_fn = FITTERS[cand.kind](
        Xt, yt, w, Xe, ye, cand.params, rounds, seed + cand.seed_offset)
    pred = np.full(len(X), np.nan)
    pred[vm] = pred_fn(X.loc[vm, keep])
    return pred, n_rounds, int(tm.sum()), time.time() - t0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(df, base, cands, n_boot=400, extra_folds=False, far=False, prov=None):
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)
    prov = prov or {}

    late = V.tail_late_fold()
    recent = V.tail_recent_fold()
    primary = V.primary_fold()
    groups = [(late.train_end, [primary, late]), (recent.train_end, [recent])]
    if far:
        # 75-91 days ahead over the same rows -- past what the real task asks,
        # which is the point. Early-stops on the 91-day window at that cutoff
        # so `tail_far` stays a read rather than a tuned-to target.
        groups.append((V.tail_far_fold().train_end,
                       [V.tail_far_fold()]))
    if extra_folds:
        for f in V.secondary_folds():
            groups.append((f.train_end, [f]))

    te_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def te_for(cutoff):
        if cutoff not in te_cache:
            te_cache[cutoff] = encoding.build(df, fit_cutoff=cutoff)
        return te_cache[cutoff]

    def X_for(cutoff):
        return pd.concat([base, te_for(cutoff)], axis=1)

    tail_m = late.val_mask(ts, is_test)
    y_tail = y[tail_m]
    # The full column universe, resolved once and through the same cache the
    # fits use, so naming the columns does not cost an extra encoder build. The
    # encoder is rebuilt per cutoff but always emits the same column names.
    all_cols = list(base.columns) + list(te_for(late.train_end).columns)

    rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    rounds_book: dict[str, dict] = {}

    for cand in cands:
        t0 = time.time()
        row = {"candidate": cand.name, "kind": cand.kind,
               "halflife": cand.halflife, "train_days": cand.train_days,
               "drop_blocks": ",".join(sorted(cand.drop_blocks or []))}
        keep = None
        if cand.drop_blocks:
            # Column *names*, not the matrix: `X_for` concatenates base and
            # encoder into a fresh ~1 GB frame every call, and asking it for a
            # column list would allocate one just to throw it away.
            keep = keep_columns(all_cols, prov, cand.drop_blocks)
            row["n_features"] = len(keep)
            if len(keep) == len(all_cols):
                raise SystemExit(
                    f"{cand.name} drops {sorted(cand.drop_blocks)} but no column "
                    "matched; the provenance stamp is missing or stale.")
        ref_rounds = ref_n = None
        try:
            for cutoff, folds in groups:
                X = X_for(cutoff)
                # Solo cutoffs borrow the reference round count, scaled by the
                # square root of the training-row ratio: more data supports a
                # longer fit, but not proportionally, and on a drifting target
                # overshooting is the more expensive mistake.
                fixed = None
                if len(folds) == 1 and ref_rounds is not None:
                    n_here = int(train_mask(ts, is_test, cutoff, cand.train_days).sum())
                    fixed = max(int(ref_rounds * (n_here / max(ref_n, 1)) ** 0.5), 50)
                pred, n_rounds, n_train, secs = fit_at(
                    X, y, ts, is_test, cutoff, folds, cand, fixed_rounds=fixed,
                    keep=keep)
                if ref_rounds is None:
                    ref_rounds, ref_n = n_rounds, n_train
                report = [f for f in folds if not f.name.startswith(("es_", "far_wide"))]
                for _, r in E.fold_scores(ts, y, pred, is_test, report).iterrows():
                    row[f"ap_{r['slice']}"] = r["ap"]
                    row[f"lift_{r['slice']}"] = r["lift"]
                row[f"rounds_{cutoff.date()}"] = n_rounds
                row[f"ntrain_{cutoff.date()}"] = n_train
                if cutoff == late.train_end:
                    preds[f"{cand.name}@late"] = pred[tail_m]
                elif cutoff == recent.train_end:
                    preds[f"{cand.name}@recent"] = pred[tail_m]
                elif far and cutoff == V.tail_far_fold().train_end:
                    preds[f"{cand.name}@far"] = pred[tail_m]
        except ImportError as e:
            print(f"  {cand.name:<14s} SKIPPED ({e})", flush=True)
            continue

        row["secs"] = round(time.time() - t0)
        # `drop_blocks` and `seed_offset` go into the book because `submit.py`
        # refits from it: a member whose feature view or seed schedule is not
        # recorded would be silently rebuilt as a different model on the real
        # test set than the one that was measured here.
        rounds_book[cand.name] = {"kind": cand.kind, "params": cand.params,
                                  "halflife": cand.halflife,
                                  "train_days": cand.train_days,
                                  "drop_blocks": sorted(cand.drop_blocks or []),
                                  "seed_offset": cand.seed_offset,
                                  "rounds": ref_rounds, "n_train": ref_n}
        rows.append(row)
        far_txt = (f"  far={row['ap_tail_far']:.4f}" if "ap_tail_far" in row else "")
        print(f"  {cand.name:<14s} tail_late={row.get('ap_tail_late', np.nan):.4f}  "
              f"tail_recent={row.get('ap_tail_recent', np.nan):.4f}{far_txt}  "
              f"primary={row.get('ap_primary_62d', np.nan):.4f}  "
              f"rounds={ref_rounds}  {row['secs']}s", flush=True)

    out = pd.DataFrame(rows)

    # Paired bootstrap against the reference, on both tail geometries. Paired
    # because the two candidates score identical rows, so the shared sampling
    # noise cancels; an unpaired comparison of two +/-0.027 bands would call
    # almost nothing significant.
    ref = cands[0].name
    deltas = []
    for cand in cands:
        rec = {"candidate": cand.name}
        for tag, label in (("late", "tail_late"), ("recent", "tail_recent"),
                           ("far", "tail_far")):
            a, b = f"{ref}@{tag}", f"{cand.name}@{tag}"
            if a not in preds or b not in preds:
                continue
            if cand.name == ref:
                rec[f"d_{label}"] = 0.0
                rec[f"lo_{label}"] = 0.0
                rec[f"hi_{label}"] = 0.0
                continue
            d = E.paired_ap_delta(y_tail, preds[a], preds[b], n_boot=n_boot)
            rec[f"d_{label}"] = d["delta"]
            rec[f"lo_{label}"] = d["lo"]
            rec[f"hi_{label}"] = d["hi"]
        deltas.append(rec)
    out = out.merge(pd.DataFrame(deltas), on="candidate", how="left")
    out["verdict"] = [verdict(r) for _, r in out.iterrows()]
    return out, preds, rounds_book


def verdict(r) -> str:
    """A candidate must clear the floor on the tail and not contradict itself.

    `tail_late` is the selection window and `tail_recent` scores the identical
    rows from a later cutoff. A change that helps one and hurts the other is
    reading fold-specific noise, and the real test window -- which spans both
    horizons -- would get whichever of the two happened to be luck.
    """
    dl, dr = r.get("d_tail_late"), r.get("d_tail_recent")
    if pd.isna(dl):
        return ""
    if dl == 0.0 and dr == 0.0:
        return "reference"
    if abs(dl) < NOISE_FLOOR and (pd.isna(dr) or abs(dr) < NOISE_FLOOR):
        return "no result (< noise floor)"
    if not pd.isna(dr) and np.sign(dl) != np.sign(dr) and \
            max(abs(dl), abs(dr)) > NOISE_FLOOR:
        return "SPLIT (folds disagree)"
    if dl > 0 and r.get("lo_tail_late", -1) > 0:
        return "BETTER"
    if dl < 0 and r.get("hi_tail_late", 1) < 0:
        return "WORSE"
    return "unclear"


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--set", dest="which", default="params",
                     choices=["params", "drift", "family", "diverse", "cat2", "all"])
    ap_.add_argument("--boot", type=int, default=400)
    ap_.add_argument("--no-gnn", action="store_true")
    ap_.add_argument("--extra-folds", action="store_true",
                     help="also score wf_mar/apr/may (stability, not selection)")
    ap_.add_argument("--far", action="store_true",
                     help="also score tail_far (75-91d ahead): a drift stress read")
    args = ap_.parse_args()

    df = get_stream()
    print("loading base features...")
    base = H.load_base(df, use_gnn=not args.no_gnn)
    print(f"  matrix {base.shape[1]} cols + encoder")

    if args.which == "all":
        seen, cands = set(), []
        for s in ("params", "drift", "family", "diverse"):
            for c in SETS[s]:
                if c.name not in seen:
                    seen.add(c.name)
                    cands.append(c)
    else:
        cands = SETS[args.which]

    # Block provenance for the members that are denied one. `ablate.py` reads
    # the same stamp; the encoder columns are absent from it by construction
    # (they are rebuilt per cutoff), and `keep_columns` keeps anything it has
    # no entry for rather than dropping it.
    prov = dict(H.base_provenance())
    if any(c.drop_blocks for c in cands) and not prov:
        raise SystemExit("no provenance stamp -- rebuild the feature cache before "
                         "running a candidate that drops a block")

    print(f"\n=== TUNE: {args.which} ({len(cands)} candidates) ===")
    print("selection = tail_late, confirmation = tail_recent\n")
    out, preds, book = run(df, base, cands, n_boot=args.boot,
                           extra_folds=args.extra_folds, far=args.far, prov=prov)

    cols = [c for c in ("candidate", "n_features", "ap_tail_late", "ap_tail_recent",
                        "ap_tail_far", "ap_primary_62d", "d_tail_late", "d_tail_recent",
                        "verdict") if c in out.columns]
    print("\n=== RESULTS (sorted by tail_late) ===")
    print(out[cols].sort_values("ap_tail_late", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Merge rather than overwrite: the three sets are run separately and each
    # is a partial view of the same question.
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        out = pd.concat([old[~old["candidate"].isin(out["candidate"])], out],
                        ignore_index=True)
    out.to_csv(RESULTS, index=False)

    frame = pd.DataFrame(preds)
    if TAIL_PREDS.exists():
        prev = pd.read_parquet(TAIL_PREDS)
        prev = prev.drop(columns=[c for c in frame.columns if c in prev.columns])
        if len(prev) == len(frame):
            frame = pd.concat([prev.drop(columns=["_y"], errors="ignore"), frame], axis=1)
    frame["_y"] = y_of(df)
    frame.to_parquet(TAIL_PREDS, index=False)
    old_book = json.loads(ROUND_BOOK.read_text()) if ROUND_BOOK.exists() else {}
    old_book.update(book)
    ROUND_BOOK.write_text(json.dumps(old_book, indent=2, default=str))
    print(f"\nwrote {RESULTS}\n      {TAIL_PREDS}\n      {ROUND_BOOK}")


def y_of(df):
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    m = V.tail_late_fold().val_mask(ts, is_test)
    return df[C.TARGET].to_numpy(dtype=float)[m]


if __name__ == "__main__":
    main()
