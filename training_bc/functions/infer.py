"""Run a trained boundary-aware model and get assembly-ready weights.

This is the piece a solver calls.  It takes a point cloud, builds the stencils,
evaluates the network, undoes the ``h`` normalisation and the canonical rotation,
and hands back weights in the global frame -- plus the normal-derivative weights
that Neumann and Robin rows need.

Row assembly, for reference:

    interior / Dirichlet-adjacent   L u_i = sum_j (u_j - u_i) w_lap_ji
    Neumann    (du/dn = q)          sum_j (u_j - u_i) w_n_ji = q
    Robin      (a u + b du/dn = g)  a u_i + b * sum_j (u_j - u_i) w_n_ji = g
    Dirichlet  (u = g)              u_i = g            (no stencil needed)

In every case the centre weight is implied: ``w_ii = -sum_j w_ji``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from functions.boundary import weights_to_global
from functions.moments import physical_scale
from functions.stencils import StencilBatcher, StencilSet

Array = NDArray[np.float64]


@torch.no_grad()
def predict_normalised(model, stencils: StencilSet, device: str = "cpu",
                       batch_size: int = 2048) -> NDArray[np.float32]:
    """Raw network output, (S, K, n_operators), still in normalised units."""
    model.eval().to(device)
    out = np.empty((len(stencils), stencils.stencil_size, len(model.operators)),
                   dtype=np.float32)
    loader = StencilBatcher(stencils, batch_size=batch_size, device=device)
    for x, edge_index, batch, sel in loader:
        pred = model(x, edge_index, batch)
        out[sel] = pred.reshape(len(sel), stencils.stencil_size, -1).cpu().numpy()
    return out


def predict_weights(model, stencils: StencilSet, device: str = "cpu",
                    batch_size: int = 2048) -> dict[str, np.ndarray]:
    """Physical weights for a stencil set.

    Returns a dict with

    ``local``
        (S, K, H) weights in the canonical frame, physical units.
    ``global``
        (S, K, H) the same, with the first-derivative heads rotated back to global
        axes.  The Laplacian column is unchanged (it is rotation invariant).
    ``normal``
        (S, K) outward normal-derivative weights, or ``None`` if the model has no
        ``x`` head.  In the canonical frame this is simply the ``x`` column.
    """
    ops: Sequence[str] = model.operators
    raw = predict_normalised(model, stencils, device=device, batch_size=batch_size)

    scale = physical_scale(ops, stencils.h)              # (S, H)
    local = raw / scale[:, None, :]

    glob = local.copy()
    if "x" in ops and "y" in ops:
        ix, iy = ops.index("x"), ops.index("y")
        gx, gy = weights_to_global(local[:, :, ix], local[:, :, iy],
                                   stencils.theta.astype(np.float64))
        glob[:, :, ix] = gx
        glob[:, :, iy] = gy

    normal = local[:, :, ops.index("x")] if "x" in ops else None
    return {"local": local, "global": glob, "normal": normal, "operators": list(ops)}


def centre_weights(w: np.ndarray) -> np.ndarray:
    """Implied centre weight ``w_ii = -sum_j w_ji`` for each stencil and operator."""
    return -w[:, 1:, :].sum(axis=1)


def apply_operator(w: np.ndarray, values: np.ndarray,
                   neighbour_index: np.ndarray) -> np.ndarray:
    """Evaluate ``sum_j (u_j - u_i) w_ji`` for every stencil.

    ``w`` is (S, K) for one operator, ``neighbour_index`` is (S, K) indices into
    ``values``, with the centre first.
    """
    u = values[neighbour_index]                 # (S, K)
    return np.sum((u - u[:, 0:1]) * w, axis=1)
