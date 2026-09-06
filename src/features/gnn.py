"""Self-supervised temporal GNN embeddings for the entity graph (Tier C).

A 2-layer GraphSAGE-style encoder over the bipartite customer/device and
customer/merchant graphs, trained per snapshot on a link-prediction objective
with negative sampling.

Two constraints shape this:

  * **No label supervision.** A GNN trained against `fraud` would be a target
    -derived feature and violate the competition's leakage rule outright. The
    objective here is purely structural: reconstruct which edges exist.
  * **Comparability across snapshots.** Each snapshot warm-starts from the
    previous snapshot's weights and embeddings, so the representation evolves
    smoothly instead of being re-randomised every week. Even so, only
    rotation-invariant quantities are exported (see graph.py).

Each snapshot is built from strictly-past edges by the caller, so the encoder
never sees an edge that postdates the rows it scores.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .. import config as C


def _row_normalise(m: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(m.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(deg, 1.0)) @ m


class _State:
    """Carried across snapshots so embeddings stay comparable over time."""

    def __init__(self, n_c: int, n_d: int, n_m: int, dim: int, seed: int):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim)
        self.h_c = rng.standard_normal((n_c, dim)).astype(np.float32) * scale
        self.h_d = rng.standard_normal((n_d, dim)).astype(np.float32) * scale
        self.h_m = rng.standard_normal((n_m, dim)).astype(np.float32) * scale
        self.W1 = np.eye(dim, dtype=np.float32) + rng.standard_normal((dim, dim)).astype(np.float32) * 0.01
        self.W2 = np.eye(dim, dtype=np.float32) + rng.standard_normal((dim, dim)).astype(np.float32) * 0.01


def train_snapshot(
    cd: sp.csr_matrix,
    cm: sp.csr_matrix,
    state: _State | None,
    seed: int,
    dim: int | None = None,
    epochs: int | None = None,
):
    """Refine embeddings on one snapshot; return (cust, dev, merch, state).

    Uses torch when available for the gradient steps, and falls back to a
    propagation-only encoder otherwise so the pipeline never hard-fails on a
    missing optional dependency.
    """
    dim = dim or C.GNN_EMBED_DIM
    epochs = epochs or C.GNN_EPOCHS
    n_c, n_d = cd.shape
    n_m = cm.shape[1]

    if state is None:
        state = _State(n_c, n_d, n_m, dim, seed)

    try:
        import torch
    except ImportError:
        return _propagate_only(cd, cm, state, dim)

    torch.manual_seed(seed)
    dev_edges = np.asarray(cd.nonzero())
    if dev_edges.shape[1] < 100:
        return _propagate_only(cd, cm, state, dim)

    # --- neighbourhood aggregation (the "SAGE" part), done in scipy --------
    A_cd = _row_normalise(cd)
    A_dc = _row_normalise(cd.T.tocsr())
    A_cm = _row_normalise(cm)
    A_mc = _row_normalise(cm.T.tocsr())

    h_c, h_d, h_m = state.h_c, state.h_d, state.h_m
    for _ in range(2):  # 2 layers
        agg_c = 0.5 * (A_cd @ h_d) + 0.5 * (A_cm @ h_m)
        agg_d = A_dc @ h_c
        agg_m = A_mc @ h_c
        h_c = np.tanh(h_c @ state.W1 + agg_c @ state.W2)
        h_d = np.tanh(h_d @ state.W1 + agg_d @ state.W2)
        h_m = np.tanh(h_m @ state.W1 + agg_m @ state.W2)

    # --- link-prediction refinement, no labels involved --------------------
    t_c = torch.tensor(h_c, requires_grad=True)
    t_d = torch.tensor(h_d, requires_grad=True)
    opt = torch.optim.Adam([t_c, t_d], lr=0.01)

    src = torch.tensor(dev_edges[0], dtype=torch.long)
    dst = torch.tensor(dev_edges[1], dtype=torch.long)
    n_edges = src.shape[0]
    batch = min(n_edges, 20_000)
    g = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        idx = torch.randint(0, n_edges, (batch,), generator=g)
        s, d = src[idx], dst[idx]
        neg = torch.randint(0, n_d, (batch,), generator=g)
        pos_score = (t_c[s] * t_d[d]).sum(1)
        neg_score = (t_c[s] * t_d[neg]).sum(1)
        loss = (
            torch.nn.functional.softplus(-pos_score).mean()
            + torch.nn.functional.softplus(neg_score).mean()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    state.h_c = t_c.detach().numpy().astype(np.float32)
    state.h_d = t_d.detach().numpy().astype(np.float32)
    state.h_m = h_m.astype(np.float32)
    return state.h_c, state.h_d, state.h_m, state


def _propagate_only(cd, cm, state: _State, dim: int):
    """Torch-free fallback: pure neighbourhood propagation, no training."""
    A_cd = _row_normalise(cd)
    A_dc = _row_normalise(cd.T.tocsr())
    A_cm = _row_normalise(cm)
    A_mc = _row_normalise(cm.T.tocsr())
    h_c, h_d, h_m = state.h_c, state.h_d, state.h_m
    for _ in range(2):
        agg_c = 0.5 * (A_cd @ h_d) + 0.5 * (A_cm @ h_m)
        h_c = np.tanh(h_c @ state.W1 + agg_c @ state.W2)
        h_d = np.tanh(h_d @ state.W1 + (A_dc @ h_c) @ state.W2)
        h_m = np.tanh(h_m @ state.W1 + (A_mc @ h_c) @ state.W2)
    state.h_c, state.h_d, state.h_m = (
        h_c.astype(np.float32), h_d.astype(np.float32), h_m.astype(np.float32)
    )
    return state.h_c, state.h_d, state.h_m, state
