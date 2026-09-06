"""How different is the test period from the training period, and where?

    .venv/Scripts/python.exe -u adversarial_validation.py            # all modes
    .venv/Scripts/python.exe -u adversarial_validation.py --mode raw

Train a classifier to tell train rows from test rows. If it cannot, the two
periods are exchangeable and drift is not your problem. If it can, its feature
importances name exactly which columns moved.

Three modes, because "AUC is high" on its own is close to meaningless here:

  `raw`     raw columns only. The clean read on genuine population drift --
            no engineered feature can confound it.
  `full`    the engineered matrix. Names the features that shifted most.
  `within`  early train (Jan-Apr) against the tail window (Jun 29 - Jul 15),
            which is the same drift measured where we still have labels, and
            therefore the only version we can act on and check.

**The trap this script is built to avoid.** The target `is_test` is a perfect
function of time, so any feature that trends with stream position separates the
two sets perfectly without any distributional change having occurred. A
lifetime counter like `cust_prior_n` only goes up; `dev_age_days` only goes up.
An adversarial model handed those reports AUC ~1.0 and tells you nothing. So:

  * `config.BANNED_FEATURES` is dropped (id, raw epoch, row order).
  * Every feature gets a `monotone_in_time` flag from the correlation of its
    weekly mean with week index, and the headline AUC is reported both with
    and without the flagged ones. The second number is the one that means
    "distribution changed".
  * The split for the adversarial model is **grouped by customer**, so it
    cannot win by memorising which customers appear on which side.

Outputs, all under `data/processed/`:

  `adversarial_features.csv`  per-feature AUC, PSI, KS, gain, drift flags
  `adversarial_weights.parquet`  per-train-row density-ratio weights, and the
                              propensity they came from, for Stage 2/3 to A/B
                              against plain recency weighting
  `adversarial_by_week.csv`   mean p(test) per training week -- how far back
                              the "test-like" region extends
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config as C
from src.io_utils import get_stream

FEAT_OUT = C.PROCESSED / "adversarial_features.csv"
WEIGHT_OUT = C.PROCESSED / "adversarial_weights.parquet"
WEEK_OUT = C.PROCESSED / "adversarial_by_week.csv"
BASE_CACHE = C.PROCESSED / "base_features.parquet"

ADV_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=0,
    verbosity=-1,
    seed=C.SEED,
)

# The familiar PSI bands -- <0.10 stable, 0.10-0.25 moderate, >0.25 significant
# -- trace back to Lewis (1994), *An Introduction to Credit Scoring*. They are
# a rule of thumb calibrated for scorecard-review samples in the hundreds, and
# Yurdakul & Naranjo show they carry no fixed error rate: PSI is asymptotically
# (1/n + 1/m) * chi2_{B-1}, so the value that means "shifted at the 5% level"
# shrinks as the samples grow.
#
# At this project's scale that gap is enormous. For two samples of ~67k rows
# and 10 bins the 5% critical value is around 0.0005, roughly 500x below the
# 0.25 band. Judged by the rule of thumb, almost nothing here is significant;
# judged by the test, almost everything is.
#
# So both are reported and neither is used alone: `psi` as an **effect size**
# for ranking which features moved most, and `psi_significant` as the honest
# pass/fail at this sample size. The bands are kept for orientation only.
PSI_STABLE, PSI_MODERATE = 0.10, 0.25
PSI_ALPHA = 0.05

# A per-feature adversarial AUC at or above this makes a feature a candidate
# for dropping. The value comes from Kim & Lee (RecSys Challenge 2023), who
# filter "variables with AUC >= 0.75 (indicating a potential covariate shift)".
# It is stated without derivation for one competition, so treat it as a
# shortlist rule and not a test -- the shortlist is written out to be A/B'd on
# the tail window, never applied here.
ADV_AUC_DROP = 0.75


# ---------------------------------------------------------------------------
# per-feature diagnostics
# ---------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a comparison sample.

        PSI = sum_b (a_b - e_b) * ln(a_b / e_b)

    over `bins` quantile buckets cut on the *expected* (reference) sample, so
    the reference is uniform across buckets by construction and the statistic
    reads as "how far did the comparison sample move off that shape". Missing
    values get their own bucket -- a change in missing rate is a real shift and
    dropping it would hide one. Empty buckets are floored to avoid ln(0).
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e_ok, a_ok = np.isfinite(e), np.isfinite(a)
    if e_ok.sum() < bins or a_ok.sum() < 1:
        return float("nan")

    edges = np.unique(np.nanquantile(e[e_ok], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_cnt = np.histogram(e[e_ok], bins=edges)[0].astype(float)
    a_cnt = np.histogram(a[a_ok], bins=edges)[0].astype(float)
    # Missing as its own bucket.
    e_cnt = np.append(e_cnt, (~e_ok).sum())
    a_cnt = np.append(a_cnt, (~a_ok).sum())

    e_pct = np.maximum(e_cnt / max(e_cnt.sum(), 1), 1e-6)
    a_pct = np.maximum(a_cnt / max(a_cnt.sum(), 1), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_critical(n: int, m: int, bins: int = 10, alpha: float = PSI_ALPHA) -> float:
    """The PSI value that means "shifted" at level `alpha` for these sizes.

    Yurdakul & Naranjo, eq. 4.1: under the null of no shift,

        PSI ~ (1/n + 1/m) * chi2_{B-1}

    so the critical value is (1/n + 1/m) * chi2_{alpha, B-1}. Their verdict on
    the rule of thumb: it "seems reasonable for sample sizes n and m between
    100 and 200, but it is too conservative for larger sample sizes".
    """
    from scipy.stats import chi2

    return float((1 / max(n, 1) + 1 / max(m, 1)) * chi2.ppf(1 - alpha, bins))


def ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, NaNs dropped."""
    from scipy.stats import ks_2samp

    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < 2 or len(a) < 2:
        return float("nan")
    return float(ks_2samp(e, a).statistic)


def monotone_in_time(v: np.ndarray, week: np.ndarray) -> float:
    """|Spearman| of the feature's weekly mean against week index.

    A lifetime counter or an age in days rises every week by construction. Such
    a feature separates train from test perfectly while telling you nothing
    about a change in behaviour, so it has to be identified and set aside
    before the adversarial AUC can be read as drift.
    """
    from scipy.stats import spearmanr

    s = pd.Series(v).groupby(week).mean()
    s = s[np.isfinite(s.to_numpy())]
    if len(s) < 5:
        return float("nan")
    r = spearmanr(np.arange(len(s)), s.to_numpy()).statistic
    return float(abs(r)) if np.isfinite(r) else float("nan")


def single_feature_auc(v: np.ndarray, label: np.ndarray) -> float:
    """AUC of one feature at separating the two sets. Missing sorts low."""
    x = np.where(np.isfinite(v), v, -1e18)
    if len(np.unique(x)) < 2:
        return 0.5
    return float(max(roc_auc_score(label, x), 1 - roc_auc_score(label, x)))


# ---------------------------------------------------------------------------
# the adversarial model
# ---------------------------------------------------------------------------
def adversarial(
    X: pd.DataFrame,
    label: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 2,
    rounds: int = 400,
) -> tuple[float, pd.Series, np.ndarray]:
    """Grouped out-of-fold adversarial classifier.

    Splitting by customer matters: a random row split lets the model recognise
    customers rather than distributions, and every entity-derived feature then
    looks like drift. The grouping makes the AUC a statement about the feature
    distributions, which is what we asked.

    Returns (OOF AUC, gain shares, OOF propensity p(row is from the later set)).
    """
    import lightgbm as lgb

    rng = np.random.default_rng(C.SEED)
    ug = pd.unique(groups)
    assign = pd.Series(rng.integers(0, n_folds, len(ug)), index=ug)
    fold_of = assign.reindex(groups).to_numpy()

    oof = np.zeros(len(X))
    gains = []
    for k in range(n_folds):
        tr, va = fold_of != k, fold_of == k
        m = lgb.train(
            ADV_PARAMS,
            lgb.Dataset(X[tr], label=label[tr]),
            num_boost_round=rounds,
            valid_sets=[lgb.Dataset(X[va], label=label[va])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        g = pd.Series(m.feature_importance("gain"), index=X.columns)
        gains.append(g / max(g.sum(), 1e-12))

    return float(roc_auc_score(label, oof)), pd.concat(gains, axis=1).mean(axis=1), oof


def diagnose(
    X: pd.DataFrame,
    label: np.ndarray,
    groups: np.ndarray,
    week: np.ndarray,
    name: str,
    top: int = 25,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Full adversarial report for one comparison."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(f"  {int((~label.astype(bool)).sum()):,} reference rows vs "
          f"{int(label.sum()):,} comparison rows, {X.shape[1]} features")

    auc, gain, oof = adversarial(X, label, groups)
    print(f"\n  adversarial AUC (all features)          {auc:.4f}")

    ref, cmp_ = ~label.astype(bool), label.astype(bool)
    rows = []
    for c in X.columns:
        v = X[c].to_numpy(dtype=float)
        rows.append(
            {
                "feature": c,
                "gain": float(gain.get(c, 0.0)),
                "auc": single_feature_auc(v, label),
                "psi": psi(v[ref], v[cmp_]),
                "ks": ks(v[ref], v[cmp_]),
                "time_monotone": monotone_in_time(v, week),
                "mean_ref": float(np.nanmean(v[ref])) if np.isfinite(v[ref]).any() else np.nan,
                "mean_cmp": float(np.nanmean(v[cmp_])) if np.isfinite(v[cmp_]).any() else np.nan,
            }
        )
    rep = pd.DataFrame(rows)
    # A position artifact has to do BOTH things: trend with stream position,
    # and actually separate the two sets. Monotonicity alone is not enough --
    # `amount_bdt` has a monotone weekly mean here (its extreme tail is
    # shrinking) yet a PSI of 0.0002 and an AUC of 0.501, so its bulk
    # distribution did not move at all and calling it an artifact would throw
    # away a real, if small, signal. Requiring separation as well leaves the
    # flag on things like `account_age_days`, which rises by one per day for
    # every returning customer and separates on that alone.
    # The AUC floor matters. At 0.52 the flag catches 93 of 187 features and
    # 40% of the fraud model's gain, because `amt_ratio_median` and friends
    # have a trending weekly mean (the fraud-amount decay drags it) while
    # separating train from test no better than a coin. Those are real signal,
    # not artifacts. At 0.60 the flag catches 44 features carrying 7.5% of
    # gain, and what it catches is unbounded lifetime counters and raw ages --
    # exactly the intended target.
    rep["position_artifact"] = (rep.time_monotone > 0.80) & (rep.auc >= 0.60)
    # Out-of-range severity. Above ~0.9 a feature's test values barely overlap
    # its train values at all: every test row lands past the largest split
    # point the trees ever saw, so the feature is effectively constant at
    # scoring time no matter how much gain it earned in training.
    rep["severity"] = pd.cut(
        rep.auc.where(rep.position_artifact),
        [-np.inf, 0.70, 0.90, np.inf],
        labels=["mild", "moderate", "severe"],
    )
    rep["psi_band"] = pd.cut(
        rep.psi, [-np.inf, PSI_STABLE, PSI_MODERATE, np.inf],
        labels=["stable", "moderate", "significant"],
    )
    crit = psi_critical(int(ref.sum()), int(cmp_.sum()))
    rep["psi_significant"] = rep.psi > crit
    # A shortlist, not a decision: features a single-feature adversarial
    # classifier can separate on, minus the ones that merely count time.
    rep["drop_candidate"] = (rep.auc >= ADV_AUC_DROP) & ~rep.position_artifact
    rep = rep.sort_values("gain", ascending=False).reset_index(drop=True)
    print(f"  PSI 5% critical value at these sample sizes: {crit:.5f} "
          f"(the 0.25 rule of thumb is {0.25 / max(crit, 1e-12):,.0f}x larger)")

    # The number that actually means "the distribution changed".
    keep = [c for c in X.columns if c not in set(rep.loc[rep.position_artifact, "feature"])]
    auc_clean = float("nan")
    if len(keep) >= 2 and len(keep) < X.shape[1]:
        auc_clean, _, _ = adversarial(X[keep], label, groups)
        print(f"  adversarial AUC (position artifacts dropped) {auc_clean:.4f}  "
              f"[{X.shape[1] - len(keep)} dropped]")
    elif len(keep) == X.shape[1]:
        auc_clean = auc
        print("  no position artifacts found; the AUC above is the clean one")

    # An assumption-free companion to the flag above: drop whichever single
    # feature the adversarial model leaned on hardest and see what is left. If
    # the AUC collapses to ~0.5, one column was carrying the entire apparent
    # drift and there is nothing else to find.
    top_feat = rep.iloc[0]["feature"]
    rest = [c for c in X.columns if c != top_feat]
    if len(rest) >= 2:
        auc_wo, _, _ = adversarial(X[rest], label, groups)
        print(f"  adversarial AUC without `{top_feat}` (top gain)  {auc_wo:.4f}")

    print(f"\n  --- top {top} by adversarial gain ---")
    cols = ["feature", "gain", "auc", "psi", "psi_band", "ks", "time_monotone",
            "position_artifact", "mean_ref", "mean_cmp"]
    print(rep[cols].head(top).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    real = rep[~rep.position_artifact].sort_values("psi", ascending=False)
    print(f"\n  --- top {top} genuine distribution shifts (PSI, artifacts excluded) ---")
    print(real[cols].head(top).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    n_sig = int((real.psi > PSI_MODERATE).sum())
    n_mod = int(((real.psi > PSI_STABLE) & (real.psi <= PSI_MODERATE)).sum())
    print(f"\n  PSI summary (non-artifact features): by the rule of thumb "
          f"{n_sig} significant (>{PSI_MODERATE}), {n_mod} moderate, "
          f"{len(real) - n_sig - n_mod} stable; by the size-aware test "
          f"{int(real.psi_significant.sum())} of {len(real)} shifted")

    cand = rep[rep.drop_candidate]
    print(f"\n  --- drop shortlist: {len(cand)} features separate train from test "
          f"on their own (AUC >= {ADV_AUC_DROP}) without being time counters ---")
    if len(cand):
        print("  " + ", ".join(cand.feature.head(40)))
        print("  Candidates only. Adversarial feature *selection* is the one shift")
        print("  correction with GBDT evidence behind it, but Pan et al. also record")
        print("  a 12-point AUC collapse from over-dropping, so each has to earn its")
        print("  removal on the tail window.")

    art = rep[rep.position_artifact].sort_values("auc", ascending=False)
    if len(art):
        print(f"\n  --- {len(art)} time-counter artifacts, worst first ---")
        print(art[["feature", "auc", "psi", "severity", "mean_ref", "mean_cmp"]].head(20).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
        print("  A feature that only counts upward has no in-range values in the test")
        print("  period: every test row sits past the largest split point the trees")
        print("  ever saw. This is the `transaction_id` failure under another name,")
        print("  and BANNED_FEATURES does not catch it.")

    rep.insert(0, "comparison", name)
    return rep, oof, auc_clean


def iterative_drop(
    X: pd.DataFrame,
    label: np.ndarray,
    groups: np.ndarray,
    auc_target: float = 0.75,
    max_iter: int = 15,
    per_iter: int = 6,
) -> pd.DataFrame:
    """Pan et al.'s loop: strip the top separators until train and test blur.

    From the AdKDD 2020 paper: train an adversarial classifier; while its AUC
    exceeds a threshold, drop the highest-importance features and refit. What
    comes back is the minimal-ish set of columns responsible for the train/test
    boundary being learnable at all.

    Two reasons this is worth the compute here. It is the **only** shift
    correction with published GBDT evidence behind it -- the same paper found
    inverse-propensity weighting losing to an untouched baseline on all six of
    its drifted datasets, while feature selection won. And it does not stop at
    the obvious counters: a matrix can stay perfectly separable after the
    obvious ones go, which is exactly what happens here, and the loop is what
    surfaces the next layer.

    It returns a drop *order*, not a decision. The same paper records a
    12-point AUC collapse on one dataset from dropping too much, so every
    feature named here still has to earn its removal on the tail window.
    """
    remaining = list(X.columns)
    rows = []
    for it in range(max_iter):
        auc, gain, _ = adversarial(X[remaining], label, groups)
        rows.append({"iter": it, "n_features": len(remaining), "auc": auc,
                     "dropped": ""})
        print(f"  iter {it:2d}  {len(remaining):3d} features  AUC={auc:.4f}", flush=True)
        if auc < auc_target or len(remaining) <= per_iter + 1:
            break
        top = gain.sort_values(ascending=False).head(per_iter).index.tolist()
        rows[-1]["dropped"] = ", ".join(top)
        print(f"           dropping: {', '.join(top)}", flush=True)
        remaining = [c for c in remaining if c not in top]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------
def raw_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["amount_bdt"] = df["amount_bdt"].astype(np.float32)
    X["log_amount"] = np.log1p(df["amount_bdt"].to_numpy()).astype(np.float32)
    X["account_age_days"] = df["account_age_days"].astype(np.float32)
    X["hour"] = df[C.TIME_COL].dt.hour.astype(np.float32)
    X["dayofweek"] = df[C.TIME_COL].dt.dayofweek.astype(np.float32)
    for c in C.LOW_CARD_CATS:
        X[c] = pd.factorize(df[c], sort=True)[0].astype(np.float32)
    return X


def full_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    if not BASE_CACHE.exists():
        print(f"  (no {BASE_CACHE.name}; run train_model.py or build_features.py first)")
        return None
    base = pd.read_parquet(BASE_CACHE)
    if len(base) != len(df):
        print(f"  (cached base features have {len(base):,} rows, stream has "
              f"{len(df):,} -- stale, skipping the full-feature mode)")
        return None
    drop = [c for c in base.columns if c in C.BANNED_FEATURES]
    return base.drop(columns=drop).astype(np.float32)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--mode", choices=["raw", "full", "within", "iterate", "all"],
                     default="all")
    ap_.add_argument("--top", type=int, default=25)
    args = ap_.parse_args()

    df = get_stream()
    ts = df[C.TIME_COL]
    is_test = df["is_test"].to_numpy()
    week = ts.dt.to_period("W").astype(str).to_numpy()
    cust = df[f"{C.CUSTOMER}_code"].to_numpy()

    reports = []
    oof_full = oof_raw = None

    if args.mode in ("raw", "all"):
        rep, oof, _ = diagnose(
            raw_matrix(df), is_test.astype(int), cust, week,
            "RAW COLUMNS: train (Jan 1 - Jul 15) vs test (Jul 16 - Sep 15)", args.top,
        )
        reports.append(rep)
        oof_raw = oof

    if args.mode in ("full", "all"):
        X = full_matrix(df)
        if X is not None:
            rep, oof, _ = diagnose(
                X, is_test.astype(int), cust, week,
                "ENGINEERED FEATURES: train vs test", args.top,
            )
            reports.append(rep)
            oof_full = oof

    if args.mode == "iterate":
        X = full_matrix(df)
        if X is None:
            raise SystemExit("iterate mode needs base_features.parquet")
        print(f"\n{'=' * 72}\nITERATIVE ADVERSARIAL FEATURE SELECTION\n{'=' * 72}")
        print("  Strip the strongest train/test separators until the two periods")
        print("  stop being distinguishable. The features named on the way down are")
        print("  the ones whose values do not exist in the test period's range.")
        hist = iterative_drop(X, is_test.astype(int), cust)
        hist.to_csv(C.PROCESSED / "adversarial_drop_order.csv", index=False)
        dropped = [f for r in hist.dropped for f in r.split(", ") if f]
        print(f"\n  {len(dropped)} features removed before the AUC target was met.")
        print(f"  wrote {C.PROCESSED / 'adversarial_drop_order.csv'}")
        return

    if args.mode in ("within", "all"):
        # The same drift, measured where labels exist. If this looks like the
        # train-vs-test comparison, the tail window is a fair stand-in for the
        # test period and selecting on it is justified.
        early = (~is_test) & (ts < pd.Timestamp("2026-05-01")).to_numpy()
        late = (~is_test) & (ts >= C.TAIL_START).to_numpy()
        sel = early | late
        rep, _, _ = diagnose(
            raw_matrix(df)[sel], late[sel].astype(int), cust[sel], week[sel],
            "RAW COLUMNS, LABELLED PERIOD ONLY: Jan-Apr vs the tail window "
            "(Jun 29 - Jul 15)", args.top,
        )
        reports.append(rep)

    # ---- how far back does "test-like" reach? ------------------------------
    # `within` mode compares two slices of the training period, so it produces
    # no train-vs-test propensity and there is nothing to weight or profile.
    src = oof_full if oof_full is not None else oof_raw
    if src is None:
        print("\n(no train-vs-test propensity in this mode; skipping the weekly "
              "profile and the density-ratio weights)")
        pd.concat(reports, ignore_index=True).to_csv(FEAT_OUT, index=False)
        print(f"wrote {FEAT_OUT}")
        return
    tr = ~is_test
    by_week = (
        pd.DataFrame({"week": week[tr], "p_test": src[tr]})
        .groupby("week").agg(n=("p_test", "size"), mean_p_test=("p_test", "mean"))
        .reset_index()
    )
    by_week["rank_from_end"] = np.arange(len(by_week))[::-1]
    print(f"\n{'=' * 72}\nHOW TEST-LIKE IS EACH TRAINING WEEK?\n{'=' * 72}")
    print("  Mean p(test) for training rows, by week. If this rises toward the")
    print("  cutoff, recency is a good proxy for test-likeness and a recency")
    print("  half-life can be read off the shape. If it is flat, it is not.")
    print(by_week.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    by_week.to_csv(WEEK_OUT, index=False)

    # ---- density-ratio weights --------------------------------------------
    # w_i = p_i / (1 - p_i), the IWERM form for covariate shift: it reweights
    # the training sample toward the density the test rows came from.
    #
    # **Emitted for completeness, and expected not to help.** Two independent
    # LightGBM studies found this exact correction losing to an untouched
    # baseline: Pan et al. (Uber, AdKDD 2020) saw inverse-propensity weighting
    # lose on all six of their drifted datasets by 1.7 to 4.5 AUC points, and
    # Qian et al. (credit scoring) measured 0.7202 against a 0.7237 baseline.
    # Both found adversarial *feature selection* helping where weighting hurt.
    #
    # There is also a reason specific to this dataset. Importance weighting
    # corrects **covariate shift**: it assumes p_train(y|x) = p_test(y|x) and
    # that only the marginal p(x) moved. The raw-column run above finds
    # essentially no marginal shift -- AUC 0.514 once the mechanical
    # account-age trend is removed, amount PSI 0.0002. What moved is
    # p(fraud | amount): fraud median amount fell 4,918 -> 944 BDT while the
    # legitimate median held at ~485. That is concept drift, and no reweighting
    # of x can correct it.
    #
    # Kept so the claim stays checkable rather than asserted. Clipped at the
    # 99th percentile; percentile clipping is folk practice, and the principled
    # bounded alternatives are AIWERM (w^lambda) and RIWERM.
    p = np.clip(src[tr], 1e-6, 1 - 1e-6)
    w = p / (1 - p)
    w = np.clip(w, 0, np.quantile(w, 0.99))
    w = w / w.mean()
    pd.DataFrame(
        {
            C.ID_COL: df.loc[tr, C.ID_COL].to_numpy(),
            C.TIME_COL: ts[tr].to_numpy(),
            "p_test": src[tr],
            "adv_weight": w.astype(np.float32),
        }
    ).to_parquet(WEIGHT_OUT, index=False)

    print(f"\ndensity-ratio weights: mean={w.mean():.3f} p50={np.median(w):.3f} "
          f"p99={np.quantile(w, 0.99):.3f} max={w.max():.3f}")
    ess = w.sum() ** 2 / (w**2).sum()
    print(f"  effective sample size {ess:,.0f} of {len(w):,} rows "
          f"({100 * ess / len(w):.1f}%) -- the cost of the reweighting")

    pd.concat(reports, ignore_index=True).to_csv(FEAT_OUT, index=False)
    print(f"\nwrote {FEAT_OUT}\n      {WEIGHT_OUT}\n      {WEEK_OUT}")


if __name__ == "__main__":
    main()
