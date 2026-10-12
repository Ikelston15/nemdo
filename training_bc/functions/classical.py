"""Classical stencil weights, for validation and for honest baselines.

The point of comparison is not "a neural network versus numerics" in the abstract.
It is: on *this* stencil, with *this* conditioning, does the learned operator land
close to what the local solve would have given, and how large are both sets of
weights?  Boundary stencils are where the two can diverge, because the local moment
system becomes badly conditioned and the exact solve answers with large, oscillatory
weights while a trained network answers with smooth, bounded ones.

Two variants are provided:

``min_norm``
    The minimum-norm solution of the consistency conditions ``A w = t``.  This is the
    plain, unweighted reference.

``weighted`` (default)
    Minimum *weighted* norm, ``w = W A^T (A W A^T)^-1 t`` with ``W`` a radial kernel.
    This is the LABFM / RBF-FD / WLS style choice: near neighbours are trusted more,
    which gives smoother and more local weights.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from functions.moments import monomial_power, target_moments

Array = NDArray[np.float64]


def moment_matrix(offsets: Array, order: int) -> Array:
    """``A`` with ``A[m, j] = dx_j^a dy_j^b / (a! b!)``; shape (n_moments, K)."""
    mp = monomial_power(order)
    a = mp[:, 0][:, None]
    b = mp[:, 1][:, None]
    inv_fact = np.array([1.0 / (math.factorial(int(i)) * math.factorial(int(j)))
                         for i, j in mp])[:, None]
    dx = offsets[None, :, 0] ** a
    dy = offsets[None, :, 1] ** b
    return inv_fact * dx * dy


def kernel_weights(offsets: Array, h: float, kind: str = "wendland") -> Array:
    """Radial trust weights ``W_jj`` used by the weighted solve."""
    r = np.linalg.norm(offsets, axis=1) / max(h, 1e-14)
    if kind == "wendland":           # Wendland C2, compact on r <= 1
        q = np.clip(1.0 - r, 0.0, None)
        return (q ** 4) * (4.0 * r + 1.0)
    if kind == "gaussian":
        return np.exp(-(2.0 * r) ** 2)
    if kind == "inverse":
        return 1.0 / (r ** 2 + 1e-3)
    if kind == "none":
        return np.ones_like(r)
    raise ValueError(f"unknown kernel '{kind}'")


def classical_weights(offsets: Array,
                      operators: Sequence[str],
                      order: int = 2,
                      kernel: str = "wendland",
                      ridge: float = 0.0) -> Array:
    """Reference weights for one stencil.  Returns (K, n_operators), physical units.

    ``offsets`` are the physical (un-normalised) neighbour displacements with the
    centre first.  The centre column of ``A`` is identically zero for every moment of
    order >= 1, so the centre weight comes out as 0, matching the ``(u_j - u_i)``
    operator form used throughout the repository.
    """
    A = moment_matrix(offsets, order)                # (n_mom, K)
    t = target_moments(operators, order).astype(np.float64)   # (n_mom, H)

    w_diag = kernel_weights(offsets, float(np.max(np.linalg.norm(offsets, axis=1))), kernel)
    AW = A * w_diag[None, :]
    G = AW @ A.T
    if ridge > 0.0:
        G = G + ridge * np.eye(G.shape[0])
    lam = np.linalg.solve(G, t) if np.linalg.cond(G) < 1e14 else np.linalg.lstsq(G, t, rcond=None)[0]
    return AW.T @ lam                                 # (K, H)


def classical_weights_batch(offsets: NDArray[np.float64],
                            operators: Sequence[str],
                            order: int = 2,
                            kernel: str = "wendland") -> NDArray[np.float64]:
    """Loop ``classical_weights`` over a stack of stencils.  (S, K, H)."""
    out = np.empty((offsets.shape[0], offsets.shape[1], len(operators)))
    for i in range(offsets.shape[0]):
        out[i] = classical_weights(offsets[i], operators, order=order, kernel=kernel)
    return out


def stencil_conditioning(offsets: Array, order: int = 2, kernel: str = "wendland") -> float:
    """Condition number of the local moment system -- the usual measure of how hard
    a stencil is.  Rises sharply once the neighbourhood becomes one-sided."""
    A = moment_matrix(offsets, order)
    w = kernel_weights(offsets, float(np.max(np.linalg.norm(offsets, axis=1))), kernel)
    G = (A * w[None, :]) @ A.T
    return float(np.linalg.cond(G))


def moment_residual(weights: Array, offsets: Array, operators: Sequence[str],
                    order: int = 2) -> Array:
    """``A w - t`` for physical weights and physical offsets.  (n_moments, H)."""
    A = moment_matrix(offsets, order)
    t = target_moments(operators, order).astype(np.float64)
    return A @ weights - t


def moment_matrix_batch(offsets: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """``A`` for a stack of stencils.  (S, n_moments, K)."""
    mp = monomial_power(order)
    a = mp[:, 0][None, :, None]
    b = mp[:, 1][None, :, None]
    inv_fact = np.array([1.0 / (math.factorial(int(i)) * math.factorial(int(j)))
                         for i, j in mp])[None, :, None]
    dx = offsets[:, None, :, 0] ** a
    dy = offsets[:, None, :, 1] ** b
    return inv_fact * dx * dy


def moment_residual_batch(weights: NDArray[np.float64],
                          offsets: NDArray[np.float64],
                          operators: Sequence[str],
                          order: int = 2) -> NDArray[np.float64]:
    """``A w - t`` for a stack of stencils.  weights (S, K, H) -> (S, n_moments, H)."""
    A = moment_matrix_batch(offsets, order)
    t = target_moments(operators, order).astype(np.float64)
    return np.einsum("smk,skh->smh", A, weights) - t[None, :, :]
