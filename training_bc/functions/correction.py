"""Shared machinery for the corrected operator: base weights, features, application.

Everything here works in **normalised** units -- offsets divided by the stencil radius
``h``, weights to be divided by ``h**q`` at the point of use. Keeping one convention
throughout is what lets the null-space projection, the base weights and the network
output be added together without a scaling bug, which would be invisible in the loss
and wrong in the solver.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from functions.classical import classical_weights_batch
from functions.infer import predict_normalised
from functions.projection import moment_matrix_torch, project_to_null_space
from functions.stencils import StencilSet

BaseKind = Literal["classical", "nemdo"]


def base_weights_normalised(stencils: StencilSet,
                            kind: BaseKind = "classical",
                            base_model=None,
                            operator: str = "laplace",
                            order: int = 2,
                            kernel: str = "wendland",
                            device: str = "cpu") -> NDArray[np.float32]:
    """Normalised base weights for one operator.  (S, K).

    ``classical`` solves the local moment system on each stencil (minimum weighted
    norm).  ``nemdo`` reads them off a trained operator, which is what you want when
    the local solve is the cost you are trying to avoid.
    """
    offs_norm = stencils.features[:, :, :2].astype(np.float64)

    if kind == "classical":
        w = classical_weights_batch(offs_norm, [operator], order=order, kernel=kernel)
        return w[:, :, 0].astype(np.float32)

    if kind == "nemdo":
        if base_model is None:
            raise ValueError("kind='nemdo' needs base_model")
        ops = list(base_model.operators)
        if operator not in ops:
            raise ValueError(f"base model has no '{operator}' head (has {ops})")
        raw = predict_normalised(base_model, stencils, device=device)
        return raw[:, :, ops.index(operator)].astype(np.float32)

    raise ValueError(f"unknown base kind '{kind}'")


def correction_features(stencils: StencilSet,
                        base_norm: NDArray[np.float32]) -> NDArray[np.float32]:
    """Stencil features with the base weight appended as a channel.  (S, K, 5).

    Columns 0 and 1 stay the normalised offsets, because the moment matrix and the
    consistency loss both read them from there.
    """
    s = np.max(np.abs(base_norm), axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return np.concatenate([stencils.features, (base_norm / s)[:, :, None]],
                          axis=2).astype(np.float32)


def apply_correction(raw_delta: Tensor,
                     features_flat: Tensor,
                     base_norm: Tensor,
                     mon_power: Tensor,
                     inv_factorial: Tensor,
                     n_graphs: int,
                     stencil_size: int,
                     ridge: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Project the raw correction and add it to the base.

    Returns ``(corrected_normalised, projected_delta)``, both (B, K).
    """
    A = moment_matrix_torch(features_flat, mon_power, inv_factorial,
                            n_graphs, stencil_size)
    delta = raw_delta.reshape(n_graphs, stencil_size)
    delta = project_to_null_space(delta, A, ridge=ridge)
    return base_norm + delta, delta


def operator_values(weights_norm: Tensor, u_stencil: Tensor, h: Tensor,
                    power: int = 2) -> Tensor:
    """Evaluate ``sum_j (u_j - u_i) w_ji`` for every (stencil, field) pair.

    ``weights_norm`` (B, K), ``u_stencil`` (B, F, K), ``h`` (B,).  Returns (B, F).
    """
    du = u_stencil - u_stencil[:, :, 0:1]
    return torch.einsum("bfk,bk->bf", du, weights_norm) / (h ** power)[:, None]


def apply_numpy(weights_norm: NDArray[np.float64], u_stencil: NDArray[np.float64],
                h: NDArray[np.float64], power: int = 2) -> NDArray[np.float64]:
    """Numpy twin of :func:`operator_values`, for evaluation."""
    du = u_stencil - u_stencil[:, :, 0:1]
    return np.einsum("bfk,bk->bf", du, weights_norm) / (h ** power)[:, None]
