"""Relationship structure over the customer / device / merchant graph.

The brief asks directly for this: "do groups of customers, devices, and
merchants form suspicious clusters that no single transaction reveals on its
own?" The supporting evidence is strong -- a device with >=16 prior distinct
customers runs at 26.8% fraud (15.2x lift), and within the fraud that the
amount and velocity rules miss entirely, "device shared by >=8 customers"
still shows 4.6x lift.

Leakage-safe design: snapshot-and-join
--------------------------------------
The graph is never built over all data. At each snapshot boundary B we build
the graph from transactions **strictly before B**, compute node quantities on
it, and join those onto the transactions in [B, next_B). Every row therefore
reads a graph that predates it, by construction rather than by argument. A
`days_since_graph_snapshot` column tells the model how stale the view is.

The graph is small -- 63,862 nodes, 119,311 unique customer-device edges -- so
weekly snapshots (38 of them) cost seconds each.

The rotation trap
-----------------
SVD singular vectors are sign- and rotation-arbitrary. Raw embedding
coordinates are therefore NOT comparable between two snapshots and behave as
noise if fed to a model directly. Everything exported here is instead
**rotation-invariant**: cosine similarities between the endpoints of the edge
actually used, and distances to the node's own previous position. Those are
stable regardless of how each SVD happened to orient its basis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .. import config as C

# Length of the recency window used for the fragmented "who is moving together
# right now" graph. One week matches the longest velocity window.
RECENT_WINDOW_D = 7


# --------------------------------------------------------------------------
# snapshot construction
# --------------------------------------------------------------------------
def _snapshot_bounds(df: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    start = df[C.TIME_COL].min().normalize()
    end = df[C.TIME_COL].max().normalize() + pd.Timedelta(days=1)
    bounds = pd.date_range(start, end, freq=freq)
    # `date_range` stops at the last anchor <= end, and the join loop below
    # only fills [bounds[i], bounds[i+1]). Without a terminal bound the
    # trailing partial period is never assigned a snapshot, so its rows keep
    # NaN for every graph column. That is invisible to both training and
    # validation here: the stream ends inside the test period, so it stranded
    # 10,455 test rows and zero train rows.
    if len(bounds) == 0 or bounds[-1] < end:
        bounds = bounds.append(pd.DatetimeIndex([end]))
    return bounds


def _build_bipartite(
    left: np.ndarray, right: np.ndarray, n_left: int, n_right: int
) -> sp.csr_matrix:
    """Unweighted customer x entity incidence matrix for one snapshot."""
    if len(left) == 0:
        return sp.csr_matrix((n_left, n_right))
    data = np.ones(len(left), dtype=np.float32)
    m = sp.coo_matrix((data, (left, right)), shape=(n_left, n_right))
    m = m.tocsr()
    m.data[:] = 1.0  # collapse repeat edges to presence
    return m


def _components(m: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, int]:
    """Connected components of the bipartite graph, returned per side.

    Also returns the component count: the two sides share one label space, so
    every bincount over them must use the same length.
    """
    n_l, n_r = m.shape
    top = sp.hstack([sp.csr_matrix((n_l, n_l)), m])
    bot = sp.hstack([m.T, sp.csr_matrix((n_r, n_r))])
    full = sp.vstack([top, bot]).tocsr()
    n_comp, labels = connected_components(full, directed=False)
    return labels[:n_l], labels[n_l:], n_comp


def _spectral(m: sp.csr_matrix, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised-adjacency truncated SVD -> (left_emb, right_emb)."""
    from scipy.sparse.linalg import svds

    n_l, n_r = m.shape
    k = min(dim, min(n_l, n_r) - 1)
    if k < 2 or m.nnz == 0:
        return np.zeros((n_l, dim), np.float32), np.zeros((n_r, dim), np.float32)

    dl = np.asarray(m.sum(axis=1)).ravel()
    dr = np.asarray(m.sum(axis=0)).ravel()
    Dl = sp.diags(1.0 / np.sqrt(np.maximum(dl, 1.0)))
    Dr = sp.diags(1.0 / np.sqrt(np.maximum(dr, 1.0)))
    norm = (Dl @ m @ Dr).astype(np.float32)

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(norm.shape))
    try:
        u, s, vt = svds(norm, k=k, v0=v0)
    except Exception:
        return np.zeros((n_l, dim), np.float32), np.zeros((n_r, dim), np.float32)

    order = np.argsort(-s)
    u, s, vt = u[:, order], s[order], vt[order]
    left = (u * s).astype(np.float32)
    right = (vt.T * s).astype(np.float32)
    if k < dim:  # pad so the array shape is stable across snapshots
        left = np.pad(left, ((0, 0), (0, dim - k)))
        right = np.pad(right, ((0, 0), (0, dim - k)))
    return left, right


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def build(df: pd.DataFrame, use_gnn: bool = True, verbose: bool = True) -> pd.DataFrame:
    cust = df[f"{C.CUSTOMER}_code"].to_numpy().astype(np.int64)
    dev = df[f"{C.DEVICE}_code"].to_numpy().astype(np.int64)
    merch = df[f"{C.MERCHANT}_code"].to_numpy().astype(np.int64)
    ts = df[C.TIME_COL]
    n_c, n_d, n_m = int(cust.max()) + 1, int(dev.max()) + 1, int(merch.max()) + 1
    n = len(df)

    bounds = _snapshot_bounds(df, C.GRAPH_SNAPSHOT_FREQ)
    pos = np.searchsorted(ts.to_numpy(), bounds.to_numpy(), side="left")

    cols = [
        "g_cust_degree_dev", "g_dev_degree_cust", "g_cust_degree_merch",
        "g_comp_size", "g_comp_n_cust", "g_comp_n_dev",
        "g_cust_2hop", "g_dev_2hop",
        "g_cd_cos", "g_cm_cos", "g_cust_emb_drift", "g_cust_emb_norm", "g_dev_emb_norm",
        "g_same_component", "days_since_graph_snapshot",
        "gr_comp_size", "gr_comp_n_cust", "gr_comp_n_dev", "gr_cust_degree_dev",
        "gr_dev_degree_cust", "gr_cust_2hop", "gr_same_component",
        "gr_dev_degree_ratio",
    ]
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in cols}

    prev_cust_emb = None
    gnn_state = None
    gnn_cols = ["gnn_cd_cos", "gnn_cd_score", "gnn_cust_emb_drift", "gnn_cm_cos"]
    if use_gnn:
        for c in gnn_cols:
            out[c] = np.full(n, np.nan, dtype=np.float32)
    prev_gnn_cust = None

    for i in range(len(bounds) - 1):
        hist_end = pos[i]          # rows strictly before this snapshot boundary
        blk_lo, blk_hi = pos[i], pos[i + 1]
        if blk_hi <= blk_lo:
            continue
        if hist_end < 500:         # not enough history to say anything
            continue

        h = slice(0, hist_end)
        cd = _build_bipartite(cust[h], dev[h], n_c, n_d)
        cm = _build_bipartite(cust[h], merch[h], n_c, n_m)

        # A lifetime graph of shared devices is one giant component (93% of
        # rows), so it says nothing. A short recency window fragments it, and
        # *that* is where "these accounts are moving together right now" lives.
        recent_lo = int(np.searchsorted(
            ts.to_numpy(), (bounds[i] - pd.Timedelta(days=RECENT_WINDOW_D)).to_numpy(), side="left"
        ))
        r = slice(recent_lo, hist_end)
        cd_recent = _build_bipartite(cust[r], dev[r], n_c, n_d)

        # --- Tier A: structural -------------------------------------------
        c_deg_d = np.asarray(cd.sum(axis=1)).ravel()
        d_deg_c = np.asarray(cd.sum(axis=0)).ravel()
        c_deg_m = np.asarray(cm.sum(axis=1)).ravel()

        lab_c, lab_d, n_comp = _components(cd)
        comp_n_cust = np.bincount(lab_c, minlength=n_comp)
        comp_n_dev = np.bincount(lab_d, minlength=n_comp)
        comp_size = comp_n_cust + comp_n_dev

        rlab_c, rlab_d, r_ncomp = _components(cd_recent)
        rcomp_n_cust = np.bincount(rlab_c, minlength=r_ncomp)
        rcomp_n_dev = np.bincount(rlab_d, minlength=r_ncomp)
        rcomp_size = rcomp_n_cust + rcomp_n_dev
        r_c_deg = np.asarray(cd_recent.sum(axis=1)).ravel()
        r_d_deg = np.asarray(cd_recent.sum(axis=0)).ravel()
        # Customers reachable through a device I shared in the last week.
        r_cust_2hop = np.asarray((cd_recent @ cd_recent.T).sum(axis=1)).ravel() - r_c_deg

        # 2-hop reach: customers sharing any device with me (and vice versa).
        cust_2hop = np.asarray((cd @ cd.T).sum(axis=1)).ravel() - c_deg_d
        dev_2hop = np.asarray((cd.T @ cd).sum(axis=1)).ravel() - d_deg_c

        # --- Tier B: spectral ---------------------------------------------
        emb_c, emb_d = _spectral(cd, C.GRAPH_EMBED_DIM, C.SEED + i)
        emb_cm, emb_m = _spectral(cm, C.GRAPH_EMBED_DIM, C.SEED + 1000 + i)

        blk = slice(blk_lo, blk_hi)
        bc, bd, bm = cust[blk], dev[blk], merch[blk]

        out["g_cust_degree_dev"][blk] = c_deg_d[bc]
        out["g_dev_degree_cust"][blk] = d_deg_c[bd]
        out["g_cust_degree_merch"][blk] = c_deg_m[bc]
        out["g_comp_size"][blk] = comp_size[lab_c[bc]]
        out["g_comp_n_cust"][blk] = comp_n_cust[lab_c[bc]]
        out["g_comp_n_dev"][blk] = comp_n_dev[lab_c[bc]]
        out["g_cust_2hop"][blk] = cust_2hop[bc]
        out["g_dev_2hop"][blk] = dev_2hop[bd]
        out["g_same_component"][blk] = (lab_c[bc] == lab_d[bd]).astype(np.float32)

        out["gr_comp_size"][blk] = rcomp_size[rlab_c[bc]]
        out["gr_comp_n_cust"][blk] = rcomp_n_cust[rlab_c[bc]]
        out["gr_comp_n_dev"][blk] = rcomp_n_dev[rlab_c[bc]]
        out["gr_cust_degree_dev"][blk] = r_c_deg[bc]
        out["gr_dev_degree_cust"][blk] = r_d_deg[bd]
        out["gr_cust_2hop"][blk] = r_cust_2hop[bc]
        out["gr_same_component"][blk] = (rlab_c[bc] == rlab_d[bd]).astype(np.float32)
        # Device fan-out this week vs over its whole life: a farm ramping up.
        out["gr_dev_degree_ratio"][blk] = r_d_deg[bd] / np.maximum(d_deg_c[bd], 1.0)

        # Rotation-invariant only -- never the raw coordinates.
        out["g_cd_cos"][blk] = _cos(emb_c[bc], emb_d[bd])
        out["g_cm_cos"][blk] = _cos(emb_cm[bc], emb_m[bm])
        out["g_cust_emb_norm"][blk] = np.linalg.norm(emb_c[bc], axis=1)
        out["g_dev_emb_norm"][blk] = np.linalg.norm(emb_d[bd], axis=1)
        if prev_cust_emb is not None:
            drift = 1.0 - _cos(emb_c, prev_cust_emb)
            out["g_cust_emb_drift"][blk] = drift[bc]
        prev_cust_emb = emb_c

        out["days_since_graph_snapshot"][blk] = (
            (ts.to_numpy()[blk] - bounds[i].to_numpy()).astype("timedelta64[s]").astype(float)
            / 86_400.0
        )

        # --- Tier C: temporal GNN -----------------------------------------
        if use_gnn:
            from .gnn import train_snapshot

            emb_gc, emb_gd, emb_gm, gnn_state = train_snapshot(
                cd, cm, gnn_state, seed=C.SEED + i
            )
            out["gnn_cd_cos"][blk] = _cos(emb_gc[bc], emb_gd[bd])
            out["gnn_cm_cos"][blk] = _cos(emb_gc[bc], emb_gm[bm])
            # "How surprising is this pairing to the graph?" -- the model's own
            # link score for the edge the transaction actually used.
            out["gnn_cd_score"][blk] = (emb_gc[bc] * emb_gd[bd]).sum(axis=1)
            if prev_gnn_cust is not None:
                out["gnn_cust_emb_drift"][blk] = (1.0 - _cos(emb_gc, prev_gnn_cust))[bc]
            prev_gnn_cust = emb_gc

        if verbose and i % 5 == 0:
            print(
                f"  snapshot {i+1}/{len(bounds)-1} {bounds[i].date()} "
                f"hist={hist_end:,} rows={blk_hi-blk_lo:,}",
                flush=True,
            )

    res = pd.DataFrame(out, index=df.index)
    # Derived ratios: absolute component size drifts as the graph accretes, so
    # give the model a share-of-graph version too.
    res["g_comp_size_frac"] = res["g_comp_size"] / np.maximum(n_c + n_d, 1)
    res["g_dev_share_of_cust"] = res["g_dev_degree_cust"] / np.maximum(
        res["g_cust_degree_dev"], 1.0
    )
    return res.astype(np.float32)
