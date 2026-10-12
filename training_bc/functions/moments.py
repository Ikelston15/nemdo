"""Multi-operator Taylor-consistency loss.

Same objective as ``training/functions/labfm_moments.py`` -- the weights must
reproduce the target derivative of every monomial up to ``approximation_order`` --
with two differences.

* Several operators are trained at once from one shared trunk, so the model emits a
  weight vector per operator.  ``d/dx`` and ``d/dy`` together span every first-order
  directional derivative, which is what makes a Neumann row assemblable (and in the
  canonical frame ``d/dx`` alone is the normal derivative).
* A small weight-norm penalty is added.  On a one-sided boundary stencil the
  consistency conditions no longer pin the weights down tightly, and without this
  the network is free to drift toward large, oscillatory solutions.  Classical
  meshfree methods get the same effect by taking the minimum-norm solve; this is the
  learned equivalent.

Moments are computed on the ``h``-normalised offsets, so a predicted weight ``w``
becomes a physical weight ``w / h^q`` with ``q`` the operator's order.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

# Which monomial moments each operator must reproduce. Keys are (power_x, power_y).
OPERATOR_SPEC: dict[str, dict[tuple[int, int], float]] = {
    "x":       {(1, 0): 1.0},
    "y":       {(0, 1): 1.0},
    "laplace": {(2, 0): 1.0, (0, 2): 1.0},
    "xx":      {(2, 0): 1.0},
    "yy":      {(0, 2): 1.0},
    "xy":      {(1, 1): 1.0},
}

# Scaling exponent q: physical weight = predicted weight / h**q
OPERATOR_ORDER: dict[str, int] = {"x": 1, "y": 1, "laplace": 2, "xx": 2, "yy": 2, "xy": 2}

DEFAULT_OPERATORS: tuple[str, ...] = ("x", "y", "laplace")


def monomial_power(order: int) -> NDArray[np.int64]:
    """Exponents of every monomial of total degree 1..order (degree 0 is implied)."""
    out = []
    for total in range(1, order + 1):
        for i in range(total + 1):
            out.append([total - i, i])
    return np.asarray(out, dtype=np.int64)


def target_moments(operators: Sequence[str], order: int) -> NDArray[np.float32]:
    """(n_moments, n_operators) matrix of required moment values."""
    mp = monomial_power(order)
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(mp)}
    t = np.zeros((len(mp), len(operators)), dtype=np.float32)
    for c, op in enumerate(operators):
        if op not in OPERATOR_SPEC:
            raise ValueError(f"unknown operator '{op}'; known: {sorted(OPERATOR_SPEC)}")
        for exponents, value in OPERATOR_SPEC[op].items():
            if exponents not in lookup:
                raise ValueError(
                    f"operator '{op}' needs monomial {exponents}, which requires "
                    f"approximation_order >= {sum(exponents)}")
            t[lookup[exponents], c] = value
    return t


def monomial_basis(order: int, device: str | torch.device = "cpu"
                   ) -> tuple[Tensor, Tensor]:
    """Exponents (2, n_moments) and 1/(a! b!) (n_moments,), precomputed once."""
    mp = monomial_power(order)
    inv_fact = np.array([1.0 / (math.factorial(a) * math.factorial(b)) for a, b in mp],
                        dtype=np.float32)
    return (torch.tensor(mp.T, dtype=torch.float32, device=device),
            torch.tensor(inv_fact, dtype=torch.float32, device=device))


def calc_moments_multihead(inputs: Tensor, outputs: Tensor, batch: Tensor,
                           mon_power: Tensor, inv_factorial: Tensor,
                           n_graphs: int) -> Tensor:
    """Moments of the predicted weights, for every operator head.

    Parameters
    ----------
    inputs : (N, F)
        Node features; columns 0 and 1 must be the normalised offsets dx/h, dy/h.
    outputs : (N, H)
        Predicted weight per node per operator head.
    batch : (N,)
        Graph id of each node.

    Returns
    -------
    (n_graphs, n_moments, H)
    """
    monomial = inv_factorial * ((inputs[:, 0, None] ** mon_power[0, :][None, :]) *
                                (inputs[:, 1, None] ** mon_power[1, :][None, :]))
    weighted = monomial.unsqueeze(-1) * outputs.unsqueeze(1)     # (N, n_mom, H)

    flat = weighted.reshape(weighted.shape[0], -1)
    acc = torch.zeros(n_graphs, flat.shape[1], dtype=flat.dtype, device=flat.device)
    acc.index_add_(0, batch, flat)
    return acc.reshape(n_graphs, weighted.shape[1], weighted.shape[2])


def consistency_loss(moments: Tensor,
                     target: Tensor,
                     outputs: Tensor,
                     batch: Tensor,
                     n_graphs: int,
                     sample_weight: Tensor | None = None,
                     weight_penalty: float = 1e-4,
                     operator_weight: Tensor | None = None
                     ) -> tuple[Tensor, Tensor]:
    """Weighted moment MSE plus a weight-norm penalty.

    ``sample_weight`` (n_graphs,) lets boundary stencils count for more than the far
    more numerous interior ones.  Returns ``(loss, per_graph_moment_mse)``; the second
    value is unweighted and is what to report.
    """
    err = (moments - target[None, :, :]) ** 2            # (B, n_mom, H)
    if operator_weight is not None:
        err = err * operator_weight[None, None, :]
    per_graph = err.mean(dim=(1, 2))                     # (B,)

    if sample_weight is None:
        loss = per_graph.mean()
    else:
        loss = (per_graph * sample_weight).sum() / sample_weight.sum()

    if weight_penalty > 0.0:
        sq = (outputs ** 2).mean(dim=1)                  # (N,)
        acc = torch.zeros(n_graphs, dtype=sq.dtype, device=sq.device)
        acc.index_add_(0, batch, sq)
        norm_per_graph = acc / float(outputs.shape[0] // max(n_graphs, 1))
        if sample_weight is None:
            loss = loss + weight_penalty * norm_per_graph.mean()
        else:
            loss = loss + weight_penalty * (
                (norm_per_graph * sample_weight).sum() / sample_weight.sum())

    return loss, per_graph.detach()


def physical_scale(operators: Sequence[str], h: NDArray[np.float32]) -> NDArray[np.float32]:
    """(S, H) divisor turning normalised weights into physical ones."""
    return np.stack([h ** OPERATOR_ORDER[op] for op in operators], axis=1)
