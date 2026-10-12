"""Solution fields to train and test a data-driven correction against.

A consistency-trained operator treats every polynomial of degree <= p as equally
important. A real PDE does not produce arbitrary polynomials -- a diffusion problem
produces smooth fields with a particular spectral content, plus boundary layers where
the geometry or the boundary condition forces them. An operator tuned to *that*
distribution can do better on it than one tuned to polynomials, at the same stencil
size, which is the only honest route to beating a classical solve on accuracy.

So the correction needs a sample of the fields you care about. Two sources:

``synthetic``
    Families with analytic Laplacians, so the target is exact and no PDE solve is
    needed. The important knob is **points per wavelength** (PPW): how well the node
    spacing resolves the field. A classical stencil and a tuned one agree when PPW is
    large and diverge when it is small, so PPW is what decides whether there is
    anything to win.

``comsol``
    Real temperature snapshots. The target Laplacian is recovered from the heat
    equation itself rather than exported directly -- see :func:`load_comsol_snapshots`.

Fields are returned as ``(u, laplacian)`` sampled at the node coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]
FieldFn = Callable[[Array], tuple[Array, Array]]


def ppw_to_wavenumber(ppw: float, spacing: float) -> float:
    """Points per wavelength -> wavenumber magnitude.  ``k = 2 pi / (PPW * s)``."""
    return 2.0 * np.pi / (max(ppw, 1e-9) * spacing)


# --------------------------------------------------------------------------------------
# Synthetic families (analytic Laplacian)
# --------------------------------------------------------------------------------------
def random_fourier(rng: np.random.Generator, spacing: float,
                   ppw_range: tuple[float, float] = (8.0, 24.0),
                   n_modes: int = 6) -> FieldFn:
    """Sum of plane waves with wavelengths in a band.

    ``u = sum_i a_i sin(k_i . x + phi_i)``,  ``lap u = -sum_i |k_i|^2 a_i sin(...)``.

    This is the field family for which the question "can a tuned stencil beat a
    Taylor-optimal one" has a clean answer: it is the same trade-off that
    dispersion-relation-preserving finite-difference schemes exploit.
    """
    ks, amps, phases = [], [], []
    for _ in range(n_modes):
        ppw = rng.uniform(*ppw_range)
        kmag = ppw_to_wavenumber(ppw, spacing)
        angle = rng.uniform(0, 2 * np.pi)
        ks.append(kmag * np.array([np.cos(angle), np.sin(angle)]))
        amps.append(rng.uniform(0.5, 1.5))
        phases.append(rng.uniform(0, 2 * np.pi))
    ks = np.asarray(ks)
    amps = np.asarray(amps)
    phases = np.asarray(phases)

    def f(p: Array) -> tuple[Array, Array]:
        arg = p @ ks.T + phases[None, :]                 # (N, n_modes)
        s = np.sin(arg)
        u = s @ amps
        k2 = np.sum(ks ** 2, axis=1)
        lap = -(s * (amps * k2)[None, :]).sum(axis=1)
        return u, lap

    return f


def gaussian_bumps(rng: np.random.Generator, spacing: float,
                   width_range: tuple[float, float] = (6.0, 20.0),
                   n_bumps: int = 4,
                   extent: float = 0.5) -> FieldFn:
    """Sum of Gaussians: localised features rather than global waves.

    For ``g = a exp(-r^2 / (2 s^2))`` in 2D, ``lap g = g (r^2/s^4 - 2/s^2)``.
    Widths are given in node spacings, so the family scales with resolution.
    """
    centres, sigmas, amps = [], [], []
    for _ in range(n_bumps):
        centres.append(rng.uniform(-extent, extent, 2))
        sigmas.append(rng.uniform(*width_range) * spacing)
        amps.append(rng.uniform(-1.5, 1.5))
    centres = np.asarray(centres)
    sigmas = np.asarray(sigmas)
    amps = np.asarray(amps)

    def f(p: Array) -> tuple[Array, Array]:
        d = p[:, None, :] - centres[None, :, :]          # (N, n_bumps, 2)
        r2 = np.sum(d ** 2, axis=2)
        g = np.exp(-r2 / (2 * sigmas[None, :] ** 2)) * amps[None, :]
        u = g.sum(axis=1)
        lap = (g * (r2 / sigmas[None, :] ** 4 - 2.0 / sigmas[None, :] ** 2)).sum(axis=1)
        return u, lap

    return f


def exponential_layer(rng: np.random.Generator, spacing: float,
                      thickness_range: tuple[float, float] = (4.0, 15.0)) -> FieldFn:
    """``u = a exp(b . x)``, ``lap u = |b|^2 u``: a steep, boundary-layer-like profile.

    Diffusion problems with a strong flux or convective condition develop exactly this
    kind of near-wall gradient, and it is where a stencil tuned for smooth interior
    fields tends to do worst.
    """
    thickness = rng.uniform(*thickness_range) * spacing
    angle = rng.uniform(0, 2 * np.pi)
    b = np.array([np.cos(angle), np.sin(angle)]) / thickness
    amp = rng.uniform(0.5, 1.5)

    def f(p: Array) -> tuple[Array, Array]:
        u = amp * np.exp(p @ b)
        return u, float(np.sum(b ** 2)) * u

    return f


def polynomial_field(rng: np.random.Generator, spacing: float,
                     degree: int = 2) -> FieldFn:
    """A polynomial of the consistency degree -- used to check exactness is retained."""
    powers = [(a, b) for total in range(degree + 1)
              for a in range(total + 1) for b in [total - a]]
    coeffs = rng.uniform(-1.0, 1.0, len(powers))

    def f(p: Array) -> tuple[Array, Array]:
        x, y = p[:, 0], p[:, 1]
        u = np.zeros(len(p))
        lap = np.zeros(len(p))
        for c, (a, b) in zip(coeffs, powers):
            u += c * x ** a * y ** b
            if a >= 2:
                lap += c * a * (a - 1) * x ** (a - 2) * y ** b
            if b >= 2:
                lap += c * b * (b - 1) * x ** a * y ** (b - 2)
        return u, lap

    return f


FAMILIES: dict[str, Callable[..., FieldFn]] = {
    "fourier": random_fourier,
    "bumps": gaussian_bumps,
    "layer": exponential_layer,
    "polynomial": polynomial_field,
}


# --------------------------------------------------------------------------------------
# Sampling onto a node set
# --------------------------------------------------------------------------------------
@dataclass
class FieldSample:
    """``n_fields`` scalar fields sampled on one node set, with exact Laplacians."""

    u: NDArray[np.float32]        # (F, N)
    laplacian: NDArray[np.float32]  # (F, N)
    description: str = ""

    def __len__(self) -> int:
        return len(self.u)


def sample_fields(coords: Array,
                  spacing: float,
                  n_fields: int,
                  families: Sequence[str] = ("fourier", "bumps", "layer"),
                  seed: int = 0,
                  normalise: bool = True,
                  **family_kwargs) -> FieldSample:
    """Draw ``n_fields`` fields, cycling through the named families.

    With ``normalise``, each field is scaled so its exact Laplacian has unit RMS. That
    keeps a steep field from dominating the loss purely because its magnitude is
    larger -- the operator error we care about is relative.
    """
    rng = np.random.default_rng(seed)
    us, laps = [], []
    for i in range(n_fields):
        name = families[i % len(families)]
        if name not in FAMILIES:
            raise ValueError(f"unknown field family '{name}'; known: {sorted(FAMILIES)}")
        kwargs = family_kwargs.get(name, {})
        fn = FAMILIES[name](rng, spacing, **kwargs)
        u, lap = fn(coords)
        if normalise:
            s = np.sqrt(np.mean(lap ** 2))
            if s > 1e-12:
                u, lap = u / s, lap / s
        us.append(u)
        laps.append(lap)
    return FieldSample(u=np.asarray(us, dtype=np.float32),
                       laplacian=np.asarray(laps, dtype=np.float32),
                       description=f"{n_fields} fields from {list(families)}")


def gather_on_stencils(sample: FieldSample,
                       neighbour_index: NDArray[np.int64]
                       ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Expand fields onto stencils.

    Returns ``u_stencil`` (S, F, K) and ``lap_target`` (S, F) at the centre nodes.
    Precomputing this costs ~45 MB at the shipped defaults and removes all gather
    work from the training loop.
    """
    u = sample.u                                   # (F, N)
    u_st = u[:, neighbour_index]                   # (F, S, K)
    u_st = np.ascontiguousarray(u_st.transpose(1, 0, 2))     # (S, F, K)
    lap = sample.laplacian[:, neighbour_index[:, 0]].T       # (S, F)
    return u_st.astype(np.float32), np.ascontiguousarray(lap).astype(np.float32)


# --------------------------------------------------------------------------------------
# COMSOL
# --------------------------------------------------------------------------------------
def load_comsol_snapshots(path: str,
                          times: Sequence[float],
                          alpha: float,
                          source: NDArray[np.float64] | float = 0.0,
                          skip_header: int = 0) -> tuple[Array, FieldSample]:
    """Read temperature snapshots and recover the Laplacian the operator must match.

    CSV layout: ``x, y, T(t0), T(t1), ...`` -- one column per exported time step, with
    ``times`` giving those times in seconds.

    COMSOL does not export the Laplacian, so it is recovered from the heat equation
    itself.  With ``u_t = alpha * lap(u) + f``,

        lap(u) = (u_t - f) / alpha

    and ``u_t`` comes from a second-order central difference in time (one-sided at the
    ends).  Two consequences worth stating plainly:

    * the target inherits the time-differencing error, so export **closely spaced**
      snapshots -- the error is O(dt^2) and this is usually the limiting factor;
    * the target is only as good as your ``alpha`` and source term, so use a case
      where both are known exactly, ideally a manufactured solution.

    For this reason a first correction is better trained on synthetic fields, with
    COMSOL used to *validate*. Switch to COMSOL-trained once the pipeline is trusted.

    This function has not been exercised against a real COMSOL export -- no file was
    available -- so check the parsed shapes before relying on it.
    """
    raw = np.genfromtxt(path, delimiter=",", skip_header=skip_header)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"{path}: expected x, y and at least one temperature column")

    coords = raw[:, :2]
    T = raw[:, 2:]                                    # (N, n_times)
    t = np.asarray(times, dtype=float)
    if len(t) != T.shape[1]:
        raise ValueError(f"{len(t)} times given but {T.shape[1]} temperature columns found")
    if len(t) < 2:
        raise ValueError("need at least two snapshots to estimate du/dt")

    dTdt = np.gradient(T, t, axis=1)                  # central inside, one-sided at ends
    f = np.asarray(source, dtype=float)
    lap = (dTdt - (f[:, None] if f.ndim else f)) / alpha

    logger.info("Loaded %d nodes x %d snapshots from %s; du/dt by central difference "
                "(dt min %.3g, max %.3g)", len(coords), T.shape[1], path,
                float(np.min(np.diff(t))), float(np.max(np.diff(t))))

    return coords, FieldSample(u=T.T.astype(np.float32),
                               laplacian=lap.T.astype(np.float32),
                               description=f"COMSOL snapshots from {path}")
