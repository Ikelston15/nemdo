"""Consistency-preserving projection for a learned correction.

This is the piece that makes a data-driven correction defensible rather than a
free-for-all.

The base operator satisfies the Taylor consistency conditions ``A w = t`` exactly (up
to training error): it differentiates every monomial of degree <= p correctly. If a
correction ``dw`` were simply added, that would be destroyed, and with it the
asymptotic convergence that makes the scheme a discretisation at all -- it would fit
coarse data nicely and then stop improving under refinement.

Instead the correction is projected onto the **null space** of ``A``:

    dw_proj = (I - A^+ A) dw        so    A (w + dw_proj) = A w = t

The corrected operator is therefore *exactly* as consistent as the base one, whatever
the network does. What is left is still a large space to work in: with ``K = 35``
neighbours and 5 second-order moment conditions, ``null(A)`` has dimension 30. The
correction spends that freedom on fitting the solutions the PDE actually produces,
rather than treating every polynomial as equally important.

**Which moments to project against is a separate choice from the base's consistency
order, and it matters more than it looks.** Constraining only the orders the base is
consistent to (here 2) pins the leading Taylor terms but leaves the *third*-order
moments free -- and for a second-order-consistent Laplacian those control the O(h)
truncation term. The classical minimum-norm solve happens to keep them small, which is
why it converges faster than its nominal order; a correction has no such incentive and
will happily spend them for accuracy at the resolution it was trained on. The result
is an operator that wins on the mesh it was tuned for and then stops improving under
refinement -- measured, not hypothetical: see ``main_eval_correction.py`` step 4.

So ``projection_order`` defaults *above* the base's consistency order. Projecting
against order-4 moments freezes the third- and fourth-order error terms at whatever
the base had, so the corrected operator inherits the base's asymptotic behaviour
exactly, and the correction is confined to what is genuinely free. The cost is
dimension: 35 - 14 = 21 instead of 30.

This also makes the comparison against classical weights a fair one. The weighted
least-squares solve is itself just *one* point in the affine set ``{w : A w = t}``
(the minimum-weighted-norm point). The correction searches the same set for a point
that does better on the solution distribution, so beating it is a statement about the
choice within the set, not about breaking the rules.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from functions.moments import monomial_basis


def moment_matrix_torch(features: Tensor, mon_power: Tensor, inv_factorial: Tensor,
                        n_graphs: int, stencil_size: int) -> Tensor:
    """``A`` for a batch of stencils, from normalised offsets.  (B, n_moments, K).

    ``features`` is the flat (B*K, F) node-feature tensor; columns 0 and 1 are the
    normalised offsets, exactly as the moment loss expects.
    """
    dx = features[:, 0, None]
    dy = features[:, 1, None]
    monomial = inv_factorial * ((dx ** mon_power[0, :][None, :]) *
                                (dy ** mon_power[1, :][None, :]))      # (B*K, n_mom)
    return monomial.reshape(n_graphs, stencil_size, -1).transpose(1, 2)


def project_to_null_space(delta: Tensor, A: Tensor, ridge: float = 1e-8) -> Tensor:
    """Remove from ``delta`` any component that would change the moments.

    Parameters
    ----------
    delta : (B, K)
        Raw per-node correction from the network.
    A : (B, n_moments, K)
        Moment matrix of each stencil.
    ridge
        Tikhonov term on ``A A^T``.  One-sided boundary stencils make this Gram matrix
        close to singular, and without it the solve amplifies noise into exactly the
        directions the projection is supposed to remove.

    Returns
    -------
    (B, K) correction with ``A @ dw_proj`` numerically zero.
    """
    # Row-normalise first. The moment rows span wildly different magnitudes -- a
    # fourth-order row carries d^4/24 on offsets in [-1, 1], so it is orders of
    # magnitude smaller than the first-order rows -- and that makes A A^T severely
    # ill-conditioned, so the solve leaks exactly the components it is meant to
    # remove. Scaling each row by a non-zero factor leaves the null space untouched,
    # so this is free.
    row_norm = A.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    A = A / row_norm

    Ad = torch.einsum("bmk,bk->bm", A, delta)                     # (B, n_mom)
    G = torch.einsum("bmk,bnk->bmn", A, A)                        # (B, n_mom, n_mom)

    eye = torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)[None]
    scale = G.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-30)[:, None, None]
    lam = torch.linalg.solve(G + ridge * scale * eye, Ad.unsqueeze(-1))   # (B, n_mom, 1)

    return delta - torch.einsum("bmk,bml->bk", A, lam)


def null_space_dimension(stencil_size: int, approximation_order: int) -> int:
    """How much freedom the correction actually has, per stencil."""
    from functions.moments import monomial_power
    return stencil_size - len(monomial_power(approximation_order))


def projector_numpy(offsets: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """``I - A^+ A`` for one stencil, for checking the torch path independently."""
    from functions.classical import moment_matrix
    A = moment_matrix(offsets, order)
    return np.eye(A.shape[1]) - np.linalg.pinv(A) @ A


def make_basis(approximation_order: int, device: str | torch.device = "cpu"):
    """Convenience re-export so callers need only import from this module."""
    return monomial_basis(approximation_order, device=device)
