"""Node-cloud generation for domains that have boundaries.

The training set used by ``training/`` consists of interior stencils taken from a
perturbed square grid: every centre node is completely surrounded by neighbours.
A node that sits *on* a wall is not like that.  Its stencil is one-sided, the local
Taylor (moment) system is far worse conditioned, and the weights an operator needs
there are much larger.  A model that never saw such a stencil extrapolates badly.

This module builds node clouds in bounded domains so those stencils exist in the
training data.  A domain is described only by a signed-distance function (SDF),
which keeps the generator general: interior nodes come from a jittered grid
filtered by the SDF, and boundary nodes are produced by projecting candidates onto
the zero level set with Newton's method and then thinning them to a target spacing.
Outward normals are the normalised SDF gradient.

Everything here is geometry only.  No PDE, no solution data, no boundary *values* --
those enter at assembly time (see README).  What the network has to learn is how to
differentiate on a one-sided point set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

Array = NDArray[np.float64]
SDF = Callable[[Array], Array]


# --------------------------------------------------------------------------------------
# Signed distance functions
# --------------------------------------------------------------------------------------
def sdf_box(half_extent=(0.5, 0.5)) -> SDF:
    """Axis-aligned rectangle centred on the origin."""
    half = np.asarray(half_extent, dtype=float)

    def f(p: Array) -> Array:
        q = np.abs(p) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    return f


def sdf_disk(radius: float = 0.5, centre=(0.0, 0.0)) -> SDF:
    c = np.asarray(centre, dtype=float)

    def f(p: Array) -> Array:
        return np.linalg.norm(p - c, axis=-1) - radius

    return f


def sdf_annulus(r_inner: float = 0.2, r_outer: float = 0.5) -> SDF:
    """Has an interior hole, so the cloud carries boundary nodes of both curvature signs."""

    def f(p: Array) -> Array:
        r = np.linalg.norm(p, axis=-1)
        return np.maximum(r_inner - r, r - r_outer)

    return f


def sdf_lshape(half_extent=(0.5, 0.5), cut=(0.25, 0.25)) -> SDF:
    """Square minus a corner block: supplies a re-entrant corner, the hardest stencil.

    ``cut`` is the size of the block removed from the ``(+x, +y)`` corner.  The block
    is centred *on* that corner so it protrudes past the domain: if its edges were
    made flush with the domain's instead, ``max(big, -cut)`` would be exactly zero
    along the shared edges and the projection would scatter spurious boundary nodes
    across a region that is not part of the domain at all.
    """
    big = sdf_box(half_extent)
    h = np.asarray(half_extent, dtype=float)
    c = np.asarray(cut, dtype=float)
    cut_centre = h

    def f(p: Array) -> Array:
        q = np.abs(p - cut_centre) - c
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        d_cut = outside + inside
        return np.maximum(big(p), -d_cut)

    return f


def sdf_blob(radius: float = 0.45, coeffs=((3, 0.12, 0.0), (5, 0.05, 1.1))) -> SDF:
    """Star-shaped domain r = R(theta).  Gives a range of boundary curvatures.

    This is only an approximate signed distance (the gradient is not unit length),
    which is fine here: the Newton projection normalises by |grad| and the outward
    normal is the normalised gradient either way.
    """

    def f(p: Array) -> Array:
        r = np.linalg.norm(p, axis=-1)
        th = np.arctan2(p[..., 1], p[..., 0])
        scale = 1.0 + sum(a * np.cos(k * th + phi) for k, a, phi in coeffs)
        return r - radius * scale

    return f


def sdf_grad(sdf: SDF, p: Array, eps: float = 1e-6) -> Array:
    """Central-difference gradient, so any SDF above can be used without an analytic normal."""
    g = np.empty_like(p)
    for d in range(p.shape[-1]):
        step = np.zeros(p.shape[-1])
        step[d] = eps
        g[..., d] = (sdf(p + step) - sdf(p - step)) / (2.0 * eps)
    return g


# --------------------------------------------------------------------------------------
# Node placement
# --------------------------------------------------------------------------------------
def _project_to_boundary(sdf: SDF, p: Array, iterations: int = 12) -> Array:
    """Newton iteration onto the zero level set: p <- p - d * grad / |grad|^2."""
    p = p.copy()
    for _ in range(iterations):
        d = sdf(p)
        g = sdf_grad(sdf, p)
        gg = np.sum(g * g, axis=-1)
        gg = np.where(gg < 1e-14, 1e-14, gg)
        p = p - (d / gg)[:, None] * g
    return p


def _thin(points: Array, min_dist: float, rng: np.random.Generator,
          fixed: Optional[Array] = None) -> Array:
    """Greedy Poisson-disk thinning, optionally keeping clear of an existing set.

    Accepted points are indexed in a KD-tree that is rebuilt periodically; points
    accepted since the last rebuild are held in a short pending list and checked by
    brute force, so no accepted pair is ever closer than ``min_dist``.
    """
    if len(points) == 0:
        return points
    order = rng.permutation(len(points))
    fixed_tree = cKDTree(fixed) if fixed is not None and len(fixed) else None

    kept: list[Array] = []
    indexed_tree: Optional[cKDTree] = None
    pending: list[Array] = []
    rebuild_every = 256

    for idx in order:
        q = points[idx]
        if fixed_tree is not None and fixed_tree.query(q)[0] < min_dist:
            continue
        if indexed_tree is not None and indexed_tree.query(q)[0] < min_dist:
            continue
        if pending:
            if np.min(np.linalg.norm(np.asarray(pending) - q, axis=1)) < min_dist:
                continue
        kept.append(q)
        pending.append(q)
        if len(pending) >= rebuild_every:
            indexed_tree = cKDTree(np.asarray(kept))
            pending = []
    return np.asarray(kept) if kept else np.empty((0, points.shape[1]))


@dataclass
class NodeCloud:
    """A point cloud plus everything the stencil builder needs to know about walls."""

    coords: Array            # (N, 2)
    is_boundary: NDArray[np.bool_]   # (N,)
    normals: Array           # (N, 2) outward unit normal; (0, 0) at interior nodes
    spacing: float           # nominal node spacing s
    name: str = "cloud"

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    def boundary_distance(self) -> Array:
        """Distance from every node to the nearest boundary node.

        Used instead of the SDF so that clouds loaded from a mesh -- where no SDF
        exists -- are treated identically to generated ones.
        """
        if not self.is_boundary.any():
            return np.full(self.n_nodes, np.inf)
        tree = cKDTree(self.coords[self.is_boundary])
        return tree.query(self.coords)[0]

    def nearest_boundary_normal(self) -> Array:
        """Outward normal of the nearest boundary node, for every node in the cloud."""
        if not self.is_boundary.any():
            return np.zeros_like(self.coords)
        bidx = np.flatnonzero(self.is_boundary)
        tree = cKDTree(self.coords[bidx])
        _, j = tree.query(self.coords)
        return self.normals[bidx[j]]


def generate_cloud(sdf: SDF,
                   spacing: float,
                   bounds: tuple[float, float, float, float] = (-0.6, 0.6, -0.6, 0.6),
                   jitter: float = 0.35,
                   boundary_spacing_factor: float = 1.0,
                   seed: int = 0,
                   name: str = "cloud") -> NodeCloud:
    """Fill the region ``sdf < 0`` with nodes and lay a node layer on ``sdf == 0``.

    Parameters
    ----------
    spacing
        Nominal interior node spacing ``s``.
    jitter
        Uniform perturbation of the interior grid, as a fraction of ``s``.  The
        default matches the disorder level used by the original NeMDO training data.
    boundary_spacing_factor
        Boundary node spacing as a multiple of ``s``.  Values below 1 give a
        refined wall layer, which is common in meshes and worth training on.
    """
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = bounds

    # --- boundary layer -------------------------------------------------------------
    # Seed candidates in a band around the zero level set, project, then thin.
    nx = max(int((x1 - x0) / (0.35 * spacing)), 8)
    ny = max(int((y1 - y0) / (0.35 * spacing)), 8)
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    cand = np.column_stack([gx.ravel(), gy.ravel()])
    band = np.abs(sdf(cand)) < 1.5 * spacing
    cand = cand[band]
    if len(cand) == 0:
        raise ValueError(f"No boundary band found for domain '{name}'; check bounds/spacing.")

    proj = _project_to_boundary(sdf, cand)
    proj = proj[np.abs(sdf(proj)) < 0.05 * spacing]       # drop failed Newton solves
    proj = proj[(proj[:, 0] > x0 - spacing) & (proj[:, 0] < x1 + spacing) &
                (proj[:, 1] > y0 - spacing) & (proj[:, 1] < y1 + spacing)]
    bnd = _thin(proj, boundary_spacing_factor * spacing, rng)

    g = sdf_grad(sdf, bnd)
    nrm = np.linalg.norm(g, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    normals = g / nrm                                      # outward: SDF grows outward

    # --- interior -------------------------------------------------------------------
    step = spacing
    ix = np.arange(x0, x1 + step, step)
    iy = np.arange(y0, y1 + step, step)
    IX, IY = np.meshgrid(ix, iy)
    pts = np.column_stack([IX.ravel(), IY.ravel()])
    pts = pts + rng.uniform(-jitter * spacing, jitter * spacing, pts.shape)
    pts = pts[sdf(pts) < -0.7 * spacing]                   # keep clear of the wall layer
    # Only remove genuinely coincident points: the jittered grid is already roughly
    # at spacing s, and thinning harder would punch holes in the interior.
    pts = _thin(pts, 0.55 * spacing, rng, fixed=bnd)

    coords = np.vstack([bnd, pts])
    is_boundary = np.zeros(len(coords), dtype=bool)
    is_boundary[: len(bnd)] = True
    all_normals = np.zeros_like(coords)
    all_normals[: len(bnd)] = normals

    return NodeCloud(coords=coords, is_boundary=is_boundary, normals=all_normals,
                     spacing=spacing, name=name)


# --------------------------------------------------------------------------------------
# A default training mix
# --------------------------------------------------------------------------------------
def default_domain_mix(spacing: float = 0.02, seed: int = 0) -> list[NodeCloud]:
    """Clouds spanning flat walls, convex and concave curvature, and a sharp corner.

    Boundary spacing is varied too, so the model sees graded wall layers rather than
    one fixed wall-to-interior ratio.
    """
    rng = np.random.default_rng(seed)
    specs = [
        ("square", sdf_box((0.5, 0.5)), (-0.6, 0.6, -0.6, 0.6), 1.0),
        ("square_fine_wall", sdf_box((0.5, 0.35)), (-0.6, 0.6, -0.45, 0.45), 0.6),
        ("disk", sdf_disk(0.5), (-0.6, 0.6, -0.6, 0.6), 1.0),
        ("annulus", sdf_annulus(0.18, 0.5), (-0.6, 0.6, -0.6, 0.6), 0.8),
        ("lshape", sdf_lshape((0.5, 0.5), (0.25, 0.25)), (-0.6, 0.6, -0.6, 0.6), 1.0),
        ("blob_a", sdf_blob(0.45, ((3, 0.12, 0.0), (5, 0.05, 1.1))), (-0.7, 0.7, -0.7, 0.7), 1.0),
        ("blob_b", sdf_blob(0.42, ((2, 0.18, 0.7), (4, 0.07, 2.3))), (-0.7, 0.7, -0.7, 0.7), 0.75),
    ]
    clouds = []
    for i, (name, sdf, bounds, bfac) in enumerate(specs):
        s = spacing * float(rng.uniform(0.85, 1.2))        # resolution diversity
        clouds.append(generate_cloud(sdf, s, bounds=bounds,
                                     boundary_spacing_factor=bfac,
                                     seed=seed + 100 * i + 1, name=name))
    return clouds


def holdout_domain(spacing: float = 0.018, seed: int = 9999) -> NodeCloud:
    """A shape kept out of training, for honest generalisation numbers."""
    sdf = sdf_blob(0.44, ((2, 0.10, 2.0), (3, 0.08, 0.4), (7, 0.03, 1.7)))
    return generate_cloud(sdf, spacing, bounds=(-0.7, 0.7, -0.7, 0.7),
                          boundary_spacing_factor=0.9, seed=seed, name="holdout_blob")
