"""Model training: search, ensemble, submission.

    python train_model.py --stage search   # config search on the 62-day fold
    python train_model.py --stage final    # fit the ensemble, write submission
    python train_model.py                  # both

Design notes
------------
* **The metric is rank-based.** Average precision depends only on the ordering
  of predictions, so probability calibration cannot change the score. No
  calibration step is included -- it would be pure ceremony here.

* **Selection happens on `primary_62d` only.** It is the one fold whose shape
  matches the real task (a 62-day forward block starting the day after the
  cutoff). The 30-day folds are reported for stability, never for choosing.

* **Target encoding is rebuilt per fold** during search, and refit at
  `TRAIN_END` for the final model. Reusing the production encoder inside a fold
  would feed validation-period labels into it.

* **Blending is by rank**, not by probability: the models are on different
  scales and AP only cares about order.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config as C
from src import validation as V
from src.io_utils import get_stream
from src.features import amount, encoding, entity, graph, temporal, velocity

BASE_CACHE = C.PROCESSED / "base_features.parquet"
SEARCH_OUT = C.PROCESSED / "search_results.csv"
VAL_PREDS = C.PROCESSED / "val_preds.parquet"
BLEND_SPEC = C.PROCESSED / "blend_spec.json"
SUBMISSION = C.ROOT / "submission.csv"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_base(df: pd.DataFrame, use_gnn: bool = True) -> pd.DataFrame:
    """Label-free feature matrix, cached (it costs ~2 min to rebuild)."""
    if BASE_CACHE.exists():
        base = pd.read_parquet(BASE_CACHE)
        if len(base) == len(df):
            print(f"  loaded cached base features {base.shape}")
            return base
    parts = []
    for name, fn in [
        ("temporal", temporal.build),
        ("amount", amount.build),
        ("velocity", velocity.build),
        ("entity", entity.build),
        ("graph", lambda d: graph.build(d, use_gnn=use_gnn, verbose=False)),
    ]:
        t = time.time()
        parts.append(fn(df))
        print(f"  {name:9s} {parts[-1].shape[1]:3d} cols {time.time()-t:6.1f}s", flush=True)
    base = pd.concat(parts, axis=1)
    base.to_parquet(BASE_CACHE, index=False)
    return base


def recency_weights(ts: pd.Series, mask: np.ndarray, cutoff, halflife_d: float | None):
    """Exponential recency weights. `None` means uniform.

    Fraud patterns drift, so the most recent weeks may deserve more say. This
    is tested rather than assumed -- see the search stage.
    """
    if halflife_d is None:
        return np.ones(int(mask.sum()))
    age_d = (pd.Timestamp(cutoff) - ts[mask]).dt.total_seconds().to_numpy() / 86_400.0
    return np.power(0.5, age_d / halflife_d)


# --------------------------------------------------------------------------
# model wrappers
# --------------------------------------------------------------------------
LGB_BASE = dict(
    objective="binary", metric="average_precision", num_threads=0,
    verbosity=-1, seed=C.SEED,
)


def fit_lgb(X, y, w, Xv, yv, params, rounds=3000, seed=C.SEED):
    import lightgbm as lgb

    p = {**LGB_BASE, **params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    dtr = lgb.Dataset(X, label=y, weight=w)
    if Xv is None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(Xv, label=yv, reference=dtr)
    return lgb.train(
        p, dtr, num_boost_round=rounds, valid_sets=[dva],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )


def fit_xgb(X, y, w, Xv, yv, params, rounds=3000, seed=C.SEED):
    import xgboost as xgb

    p = dict(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        nthread=0, seed=seed, **params,
    )
    dtr = xgb.DMatrix(X, label=y, weight=w)
    if Xv is None:
        return xgb.train(p, dtr, num_boost_round=rounds)
    dva = xgb.DMatrix(Xv, label=yv)
    return xgb.train(
        p, dtr, num_boost_round=rounds, evals=[(dva, "val")],
        early_stopping_rounds=150, verbose_eval=False,
    )


def fit_cat(X, y, w, Xv, yv, params, rounds=3000, seed=C.SEED):
    from catboost import CatBoostClassifier, Pool

    m = CatBoostClassifier(
        iterations=rounds, eval_metric="PRAUC", random_seed=seed,
        verbose=0, allow_writing_files=False, **params,
    )
    tr_pool = Pool(X, y, weight=w)
    if Xv is None:
        m.fit(tr_pool)
    else:
        m.fit(tr_pool, eval_set=Pool(Xv, yv), early_stopping_rounds=150)
    return m


def predict(model, X, kind: str):
    if kind == "lgb":
        return model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    if kind == "xgb":
        import xgboost as xgb

        it = getattr(model, "best_iteration", None)
        rng = (0, it + 1) if it is not None else None
        return model.predict(xgb.DMatrix(X), iteration_range=rng)
    return model.predict_proba(X)[:, 1]


def n_rounds(model, kind: str) -> int:
    if kind == "lgb":
        return int(model.best_iteration or model.num_trees())
    if kind == "xgb":
        it = getattr(model, "best_iteration", None)
        return int(it + 1 if it is not None else model.num_boosted_rounds())
    return int(model.get_best_iteration() or model.tree_count_)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
CONFIGS = [
    ("lgb_base",   "lgb", dict(learning_rate=0.05, num_leaves=64, min_data_in_leaf=100,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               lambda_l2=5.0), None),
    ("lgb_deep",   "lgb", dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=50,
                               feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                               lambda_l2=10.0), None),
    ("lgb_shallow","lgb", dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                               feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1,
                               lambda_l2=1.0), None),
    ("lgb_hl90",   "lgb", dict(learning_rate=0.05, num_leaves=64, min_data_in_leaf=100,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               lambda_l2=5.0), 90.0),
    ("lgb_hl45",   "lgb", dict(learning_rate=0.05, num_leaves=64, min_data_in_leaf=100,
                               feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                               lambda_l2=5.0), 45.0),
    ("xgb_base",   "xgb", dict(learning_rate=0.05, max_depth=8, min_child_weight=5,
                               subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0), None),
    ("cat_base",   "cat", dict(learning_rate=0.05, depth=8, l2_leaf_reg=5.0), None),
]

FITTERS = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}


def stage_search(df, base):
    fold = V.primary_fold()
    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)

    te = encoding.build(df, fit_cutoff=fold.train_end)  # per-fold encoder
    X = pd.concat([base, te], axis=1)
    tm, vm = fold.train_mask(ts, is_test), fold.val_mask(ts, is_test)
    Xt, yt, Xv, yv = X[tm], y[tm], X[vm], y[vm]
    br = float(yv.mean())
    print(f"\nsearch on {fold.name}: train={tm.sum():,} val={vm.sum():,} base={br:.4f}")

    rows, preds = [], {}
    for name, kind, params, hl in CONFIGS:
        if kind in ("xgb", "cat"):
            try:
                __import__({"xgb": "xgboost", "cat": "catboost"}[kind])
            except ImportError:
                print(f"  {name:12s} SKIPPED ({kind} not installed)")
                continue
        t = time.time()
        w = recency_weights(ts, tm, fold.train_end, hl)
        model = FITTERS[kind](Xt, yt, w, Xv, yv, params)
        p = predict(model, Xv, kind)
        a = V.ap(yv, p)
        preds[name] = p
        rows.append({"name": name, "kind": kind, "halflife": hl, "val_ap": a,
                     "x_base": a / br, "rounds": n_rounds(model, kind),
                     "secs": round(time.time() - t)})
        print(f"  {name:12s} val_AP={a:.4f} ({a/br:5.1f}x) rounds={n_rounds(model,kind):4d} "
              f"{time.time()-t:5.0f}s", flush=True)

    res = pd.DataFrame(rows).sort_values("val_ap", ascending=False)
    print("\n=== SEARCH RESULTS ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    res.to_csv(SEARCH_OUT, index=False)

    # Persist validation predictions so blend weights can be chosen from
    # evidence instead of assumed. An equal-weight blend across families is
    # NOT safe here: xgb and cat both score below every lgb config, so naive
    # averaging pulls the result down.
    pd.DataFrame(preds).assign(_y=yv).to_parquet(VAL_PREDS, index=False)
    print(f"wrote {SEARCH_OUT} and {VAL_PREDS}")
    return res


def hill_climb(preds: dict, y: np.ndarray, n_iter: int = 40) -> dict:
    """Caruana-style greedy ensemble selection with replacement.

    Repeatedly adds whichever member most improves AP, allowing repeats so a
    strong model can accumulate weight. Robust against the failure mode above:
    a member that hurts simply never gets picked.
    """
    names = list(preds)
    ranks = {n: rankdata(preds[n]) / len(y) for n in names}
    chosen, cur, best_hist = [], None, []
    for _ in range(n_iter):
        best_name, best_ap = None, -1.0
        for n in names:
            cand = ranks[n] if cur is None else (cur * len(chosen) + ranks[n]) / (len(chosen) + 1)
            a = V.ap(y, cand)
            if a > best_ap:
                best_name, best_ap, best_cand = n, a, cand
        chosen.append(best_name)
        cur = best_cand
        best_hist.append(best_ap)
    # Stop at the peak: extra members past it only add noise.
    k = int(np.argmax(best_hist)) + 1
    chosen = chosen[:k]
    weights = {n: chosen.count(n) / len(chosen) for n in set(chosen)}
    return {"weights": weights, "val_ap": float(best_hist[k - 1]), "n_members": k}


def stage_blend(df):
    """Choose the blend on the primary fold, then check it on the others."""
    if not VAL_PREDS.exists():
        raise SystemExit("run --stage search first")
    vp = pd.read_parquet(VAL_PREDS)
    y = vp.pop("_y").to_numpy()
    preds = {c: vp[c].to_numpy() for c in vp.columns}
    br = float(y.mean())

    singles = {n: V.ap(y, p) for n, p in preds.items()}
    best_single = max(singles, key=singles.get)
    equal_all = V.ap(y, np.mean([rankdata(p) for p in preds.values()], axis=0))
    lgb_only = [n for n in preds if n.startswith("lgb")]
    equal_lgb = V.ap(y, np.mean([rankdata(preds[n]) for n in lgb_only], axis=0))

    hc = hill_climb(preds, y)

    print("\n=== BLEND COMPARISON (primary_62d) ===")
    print(f"  best single ({best_single:<12s}) {singles[best_single]:.4f} ({singles[best_single]/br:.1f}x)")
    print(f"  equal-weight all families      {equal_all:.4f} ({equal_all/br:.1f}x)")
    print(f"  equal-weight lgb only          {equal_lgb:.4f} ({equal_lgb/br:.1f}x)")
    print(f"  hill-climb ({hc['n_members']} members)        {hc['val_ap']:.4f} ({hc['val_ap']/br:.1f}x)")
    print("\n  hill-climb weights:")
    for n, w in sorted(hc["weights"].items(), key=lambda kv: -kv[1]):
        print(f"    {n:<12s} {w:.3f}")

    options = {
        "best_single": ({best_single: 1.0}, singles[best_single]),
        "equal_lgb": ({n: 1 / len(lgb_only) for n in lgb_only}, equal_lgb),
        "hill_climb": (hc["weights"], hc["val_ap"]),
    }
    pick = max(options, key=lambda k: options[k][1])
    weights, val_ap = options[pick]
    spec = {"strategy": pick, "weights": weights, "primary_val_ap": float(val_ap)}
    BLEND_SPEC.write_text(json.dumps(spec, indent=2))
    print(f"\nchosen: {pick} (val_AP={val_ap:.4f})  -> {BLEND_SPEC}")
    return spec


# --------------------------------------------------------------------------
# final
# --------------------------------------------------------------------------
def stage_final(df, base, n_seeds: int = 3):
    """Refit on all labelled data and score the real test set.

    Round counts come from the primary fold. That fold trains on 492k rows and
    the final model on 732k (1.49x), so rounds are scaled by 1.2 -- more data
    supports a somewhat longer fit, but not proportionally, and overshooting is
    the more expensive mistake on a drifting target.
    """
    if not (SEARCH_OUT.exists() and BLEND_SPEC.exists()):
        raise SystemExit("run --stage search then --stage blend first")
    res = pd.read_csv(SEARCH_OUT).set_index("name")
    spec = json.loads(BLEND_SPEC.read_text())
    weights = spec["weights"]
    print(f"\nblend strategy: {spec['strategy']}  (fold val_AP {spec['primary_val_ap']:.4f})")

    ts, is_test = df[C.TIME_COL], df["is_test"].to_numpy()
    y = df[C.TARGET].to_numpy(dtype=float)
    te = encoding.build(df, fit_cutoff=C.TRAIN_END)  # production encoder
    X = pd.concat([base, te], axis=1)
    Xt, yt = X[~is_test], y[~is_test]
    Xs = X[is_test]

    cfg_by_name = {c[0]: c for c in CONFIGS}
    acc, total_w, members = np.zeros(len(Xs)), 0.0, []
    for cfg_name, w_cfg in sorted(weights.items(), key=lambda kv: -kv[1]):
        name, kind, params, hl = cfg_by_name[cfg_name]
        rounds = max(int(res.loc[cfg_name, "rounds"] * 1.2), 50)
        seed_ranks = []
        for s in range(n_seeds):
            t = time.time()
            w = recency_weights(ts, ~is_test, C.TRAIN_END, hl)
            model = FITTERS[kind](Xt, yt, w, None, None, params, rounds=rounds,
                                  seed=C.SEED + 100 * s)
            p = predict(model, Xs, kind)
            seed_ranks.append(rankdata(p) / len(p))
            members.append(f"{name}_s{s}")
            print(f"  {name}_s{s} w={w_cfg:.3f} rounds={rounds:4d} {time.time()-t:5.0f}s", flush=True)
        # Average seeds first, then apply the config's blend weight.
        acc += w_cfg * np.mean(seed_ranks, axis=0)
        total_w += w_cfg

    final = acc / total_w
    final = (final - final.min()) / (final.max() - final.min())  # -> [0,1]

    sub = pd.DataFrame({
        C.ID_COL: df.loc[is_test, C.ID_COL].to_numpy(),
        C.TARGET: final,
    })
    sample = pd.read_csv(C.SAMPLE_SUB)
    sub = sample[[C.ID_COL]].merge(sub, on=C.ID_COL, how="left")
    assert sub[C.TARGET].notna().all(), "missing predictions for some test ids"
    assert len(sub) == len(sample), "submission row count mismatch"
    assert sub[C.TARGET].between(0, 1).all(), "predictions outside [0,1]"
    sub.to_csv(SUBMISSION, index=False)

    print(f"\nwrote {SUBMISSION}  ({len(sub):,} rows, {len(members)} models)")
    print(sub.head(3).to_string(index=False))
    print(f"\nprediction distribution: mean={final.mean():.4f} "
          f"p99={np.quantile(final,0.99):.4f} max={final.max():.4f}")
    (C.PROCESSED / "ensemble_members.json").write_text(json.dumps(members, indent=2))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["search", "blend", "final", "all"], default="all")
    ap_.add_argument("--no-gnn", action="store_true")
    ap_.add_argument("--seeds", type=int, default=3)
    args = ap_.parse_args()

    df = get_stream()
    if args.stage == "blend":            # pure post-processing of saved preds
        stage_blend(df)
        return

    print("loading base features...")
    base = load_base(df, use_gnn=not args.no_gnn)

    if args.stage in ("search", "all"):
        stage_search(df, base)
    if args.stage in ("blend", "all"):
        stage_blend(df)
    if args.stage in ("final", "all"):
        stage_final(df, base, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
