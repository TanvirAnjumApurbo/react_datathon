"""Fit-and-score over folds, without re-fitting what can be shared.

Two facts shape this module.

**Folds that share a cutoff share a model.** `primary_62d` and `tail_late` both
train to 2026-05-14; they differ only in which rows they score. Fitting twice
would burn minutes and -- worse -- give two different models because LightGBM
here is not bit-deterministic, so the tail number would not be a clean read on
the same candidate. So folds are grouped by `train_end`, one fit per group.

**The target encoder is part of the fold, not part of the features.**
`encoding.build` defaults to `fit_cutoff=TRAIN_END`, which is right for scoring
the real test set and *wrong* inside a backtest: it would feed the fold's own
validation labels into the encoder. The grouping handles this too -- one
encoder per cutoff, rebuilt, never reused across cutoffs.

Early stopping deserves a note, and it used to be wrong. Stopping on the window
you then report is selection on that window, so the rule was "stop on the widest
window at this cutoff and report the narrow ones as free reads". That is only
honest when the narrow window sits *outside* the wide one. Here it did not:
`tail_late` (06-29 -> 07-15) is a subset of `primary_62d` (05-15 -> 07-15), so
stopping on `primary_62d` chose the round count from 1,068 of the tail's own
positives. `validation.stopping_fold` now prefers a registered window disjoint
from everything being reported (`es_late`, 05-15 -> 06-28) and only falls back
to the widest when none exists.

Where a cutoff has only the window being scored (`tail_recent`), there is no
uncontaminated choice available at all -- the stream ends inside the window --
so the round count is taken from another fit instead, the same convention
`submit.py` uses for the real submission.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path as pathlib_Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import validation as V


# ---------------------------------------------------------------------------
# the label-free matrix, cached against the source it was built from
# ---------------------------------------------------------------------------
BASE_CACHE = C.PROCESSED / "base_features.parquet"
BASE_STAMP = C.PROCESSED / "base_features.stamp.json"

_BLOCK_MODULES = ("temporal", "amount", "velocity", "entity", "behaviour",
                  "graph", "gnn", "_windows")


def feature_source_hash() -> str:
    """Digest of every feature module's source, plus the config that drives it.

    The cached matrix costs about two minutes to rebuild and `graph.build` is
    most of it, so caching is worth having. The failure it invites is worse
    than the saving: edit a feature module, forget to delete the parquet, and
    every downstream number is computed on the previous feature set with
    nothing to indicate it. That has to be a documented manual step only if
    the machine cannot do it -- and it can.

    Config values that change feature *values* are folded in as well, since
    editing a window list changes the matrix without touching a module.
    """
    h = hashlib.sha256()
    here = pathlib_Path(__file__).parent / "features"
    for name in _BLOCK_MODULES:
        f = here / f"{name}.py"
        if f.exists():
            h.update(f.read_bytes())
    h.update(
        json.dumps(
            {
                "WINDOWS_S": C.WINDOWS_S,
                "RFM_WINDOWS_S": getattr(C, "RFM_WINDOWS_S", None),
                "AMOUNT_RANK_WINDOWS_D": C.AMOUNT_RANK_WINDOWS_D,
                "GRAPH_SNAPSHOT_FREQ": C.GRAPH_SNAPSHOT_FREQ,
                "GRAPH_EMBED_DIM": C.GRAPH_EMBED_DIM,
                "GNN_EMBED_DIM": C.GNN_EMBED_DIM,
                "GNN_EPOCHS": C.GNN_EPOCHS,
                "USE_SIGNUP_INCONSISTENCY": C.USE_SIGNUP_INCONSISTENCY,
            },
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()[:16]


def load_base(df: pd.DataFrame, use_gnn: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Build (or reuse) the label-free feature matrix.

    Reuses the cache only when the row count *and* the source digest match, so
    a feature-module edit invalidates it automatically instead of relying on
    someone remembering to delete the file.
    """
    from .features import amount, entity, graph, temporal, velocity

    try:
        from .features import behaviour
    except ImportError:
        behaviour = None

    want = feature_source_hash()
    if BASE_CACHE.exists() and BASE_STAMP.exists():
        stamp = json.loads(BASE_STAMP.read_text())
        if stamp.get("hash") == want and stamp.get("n_rows") == len(df) and \
                stamp.get("use_gnn") == use_gnn:
            base = pd.read_parquet(BASE_CACHE)
            if len(base) == len(df):
                if verbose:
                    print(f"  cached base features {base.shape} (source {want})")
                return base
        elif verbose:
            print(f"  cache stale (source {stamp.get('hash')} != {want}); rebuilding")

    blocks = [("temporal", temporal.build), ("amount", amount.build),
              ("velocity", velocity.build), ("entity", entity.build)]
    if behaviour is not None:
        blocks.append(("behaviour", behaviour.build))
    blocks.append(("graph", lambda d: graph.build(d, use_gnn=use_gnn, verbose=False)))

    parts, provenance = [], {}
    for name, fn in blocks:
        t = time.time()
        blk = fn(df)
        parts.append(blk)
        sub = blk.attrs.get("subblock", {})
        for c in blk.columns:
            # A module may tag its columns with sub-blocks; use them so an
            # ablation can drop one idea rather than a whole module.
            provenance[c] = f"{name}:{sub[c]}" if c in sub else name
        if verbose:
            print(f"  {name:9s} {blk.shape[1]:3d} cols {time.time() - t:6.1f}s", flush=True)

    base = pd.concat(parts, axis=1)
    base.to_parquet(BASE_CACHE, index=False)
    BASE_STAMP.write_text(json.dumps(
        {"hash": want, "n_rows": len(base), "use_gnn": use_gnn,
         "provenance": provenance}, indent=2))
    return base


def base_provenance() -> dict[str, str]:
    """Which block each cached column came from. Empty if never built here."""
    if BASE_STAMP.exists():
        return json.loads(BASE_STAMP.read_text()).get("provenance", {})
    return {}

LGB_DEFAULTS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=64,
    min_data_in_leaf=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=0,
    verbosity=-1,
    seed=C.SEED,
)


@dataclass
class FoldFit:
    """One trained model plus its full-stream prediction vector.

    `pred` is NaN everywhere the model was not asked to predict, so it can be
    handed straight to `evaluate.fold_scores` without index bookkeeping.
    """

    cutoff: pd.Timestamp
    folds: list[V.Fold]
    pred: np.ndarray
    rounds: int
    secs: float
    gain: pd.Series | None = None
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)


def group_by_cutoff(folds: list[V.Fold]) -> dict[pd.Timestamp, list[V.Fold]]:
    """Folds keyed by training cutoff, so each cutoff is fitted once."""
    groups: dict[pd.Timestamp, list[V.Fold]] = {}
    for f in folds:
        groups.setdefault(f.train_end, []).append(f)
    return dict(sorted(groups.items()))


def _widest(folds: list[V.Fold]) -> V.Fold:
    return max(folds, key=lambda f: f.horizon_days)


def fit_cutoff(
    X: pd.DataFrame,
    y: np.ndarray,
    ts: pd.Series,
    is_test: np.ndarray,
    folds: list[V.Fold],
    params: dict | None = None,
    max_rounds: int = 2000,
    fixed_rounds: int | None = None,
    sample_weight: np.ndarray | None = None,
    categorical: list[str] | None = None,
    want_gain: bool = False,
) -> FoldFit:
    """Fit once at a shared cutoff, predict the union of the val windows."""
    import lightgbm as lgb

    p = {**LGB_DEFAULTS, **(params or {})}
    cutoff = folds[0].train_end
    assert all(f.train_end == cutoff for f in folds), "folds do not share a cutoff"

    tm = folds[0].train_mask(ts, is_test)
    vm = np.zeros(len(X), dtype=bool)
    for f in folds:
        vm |= f.val_mask(ts, is_test)

    w = None if sample_weight is None else np.asarray(sample_weight)[tm]
    t0 = time.time()
    dtr = lgb.Dataset(X[tm], label=y[tm], weight=w, categorical_feature=categorical or "auto")

    if fixed_rounds is not None:
        model = lgb.train(p, dtr, num_boost_round=fixed_rounds)
        rounds = fixed_rounds
    else:
        # Stop on a window that is *disjoint* from the ones being reported.
        # The old rule -- stop on the widest window here -- picked
        # `primary_62d`, which contains `tail_late` outright, so the round
        # count was chosen using 1,068 of the tail's own positives.
        es = V.stopping_fold(folds)
        em = es.val_mask(ts, is_test)
        dva = lgb.Dataset(X[em], label=y[em], reference=dtr)
        model = lgb.train(
            p,
            dtr,
            num_boost_round=max_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        rounds = int(model.best_iteration or model.num_trees())

    pred = np.full(len(X), np.nan)
    pred[vm] = model.predict(X[vm], num_iteration=model.best_iteration)
    secs = time.time() - t0

    gain = None
    if want_gain:
        g = pd.Series(model.feature_importance("gain"), index=X.columns)
        gain = g / max(g.sum(), 1e-12)

    fit = FoldFit(cutoff=cutoff, folds=folds, pred=pred, rounds=rounds, secs=secs, gain=gain)
    fit.scores = E.fold_scores(ts, y, pred, is_test, folds)
    return fit


def run_folds(
    build_X,
    y: np.ndarray,
    ts: pd.Series,
    is_test: np.ndarray,
    folds: list[V.Fold] | None = None,
    params: dict | None = None,
    label: str = "",
    verbose: bool = True,
    **kw,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, FoldFit]]:
    """Fit every distinct cutoff once and score all its windows.

    `build_X(cutoff) -> DataFrame` is a callback rather than a plain matrix so
    the caller can rebuild the target encoder against each fold's own cutoff.
    Anything label-free should be built once outside and closed over.

    Folds whose cutoff has no wider companion window get `fixed_rounds` from
    the first cutoff that did, rather than early-stopping on the window they
    are about to be judged on.
    """
    folds = folds or V.all_folds()
    groups = group_by_cutoff(folds)
    fits: dict[pd.Timestamp, FoldFit] = {}
    rows: list[dict] = []
    learned_rounds: int | None = None
    learned_rounds_n: int = 0

    # Cutoffs with more than one window can early-stop on the widest of them.
    # Do those first, so a solo cutoff has a round count to borrow.
    order = sorted(groups.items(), key=lambda kv: (len(kv[1]) == 1, kv[0]))

    for cutoff, grp in order:
        n_here = int(grp[0].train_mask(ts, is_test).sum())
        fixed = None
        if len(grp) == 1 and learned_rounds is not None:
            # Scale by the training-row ratio, then damp: more data supports a
            # longer fit, but not proportionally, and on a drifting target
            # overshooting is the more expensive mistake.
            fixed = max(int(learned_rounds * (n_here / max(learned_rounds_n, 1)) ** 0.5), 50)

        fit = fit_cutoff(
            build_X(cutoff), y, ts, is_test, grp, params=params, fixed_rounds=fixed, **kw
        )
        fits[cutoff] = fit
        if fixed is None:
            learned_rounds, learned_rounds_n = fit.rounds, n_here
        for _, r in fit.scores.iterrows():
            rows.append({**r.to_dict(), "rounds": fit.rounds, "secs": round(fit.secs)})
        if verbose:
            for _, r in fit.scores.iterrows():
                print(
                    f"  {label:<16s} {r['slice']:<12s} AP={r['ap']:.4f} "
                    f"({r['lift']:5.1f}x) rounds={fit.rounds:4d} {fit.secs:5.0f}s",
                    flush=True,
                )

    res = pd.DataFrame(rows)
    if label:
        res.insert(0, "candidate", label)
    return res, fits
