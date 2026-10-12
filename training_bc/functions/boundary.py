"""Boundary detection, outward normals, and the canonical (normal-aligned) frame.

Two jobs here.

1. For a point cloud that arrives without labels -- a mesh export, say -- work out
   which nodes are on a wall and which way the wall faces.  The test is neighbour
   asymmetry: at an interior node the neighbour offsets cancel, at a wall node they
   all point inwards, so the normalised mean offset is a usable outward normal.

2. Put every stencil into a canonical frame in which the outward normal is ``+x``.
   This does two useful things at once.  It removes orientation as something the
   network has to learn (the original NeMDO is not rotation-equivariant and relies
   on seeing many orientations during training), and it means the ``d/dx`` head
   *is* the normal derivative at a boundary node -- which is exactly the operator a
   Neumann or Robin row needs, with no separate model.

   Weights come back to the global frame by rotating the first-derivative pair; the
   Laplacian is rotation invariant and needs no correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

Array = NDArray[np.float64]


def estimate_boundary_and_normals(coords: Array,
                                  k: int = 12,
                                  asymmetry_threshold: float = 0.35
                                  ) -> tuple[NDArray[np.bool_], Array, Array]:
    """Label boundary nodes and estimate outward normals from geometry alone.

    For node ``i`` with ``k`` nearest neighbours at offsets ``d_j``, the asymmetry

        a_i = | mean_j d_j | / mean_j |d_j|

    is near zero when the node is surrounded and approaches ~0.5 on a straight wall.
    The outward direction is ``-mean_j d_j`` (neighbours sit on the inside).

    Returns
    -------
    is_boundary, normals, asymmetry
    """
    tree = cKDTree(coords)
    k_eff = min(k + 1, len(coords))
    _, idx = tree.query(coords, k=k_eff)
    nb = coords[idx[:, 1:]] - coords[:, None, :]          # drop self

    mean_off = nb.mean(axis=1)
    mean_dist = np.linalg.norm(nb, axis=2).mean(axis=1)
    mean_dist = np.where(mean_dist < 1e-14, 1e-14, mean_dist)
    asymmetry = np.linalg.norm(mean_off, axis=1) / mean_dist

    is_boundary = asymmetry > asymmetry_threshold

    normals = -mean_off
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-14, 1.0, nrm)
    normals = normals / nrm
    normals[~is_boundary] = 0.0
    return is_boundary, normals, asymmetry


def boundary_proximity(distance_to_boundary: Array, h: Array) -> Array:
    """Smooth 1 -> 0 flag: 1 on the wall, 0 at a full stencil radius inside.

    Preferred over a binary interior/boundary flag because the stencil of a node one
    or two layers in is also partly one-sided, and the network benefits from knowing
    how close it is rather than being told it is simply "interior".
    """
    h = np.where(h < 1e-14, 1e-14, h)
    return np.clip(1.0 - distance_to_boundary / h, 0.0, 1.0)


def canonical_angle(normals: Array, proximity: Array,
                    fallback: Array | None = None) -> Array:
    """Rotation angle theta per node, mapping the outward normal onto the local +x axis.

    Nodes with no meaningful normal (deep interior, proximity 0) get theta = 0, i.e.
    they stay in the global frame, matching how the original model was trained.
    """
    theta = np.arctan2(normals[:, 1], normals[:, 0])
    use = proximity > 0.0
    if fallback is None:
        fallback = np.zeros(len(normals))
    return np.where(use, theta, fallback)


def rotation_matrices(theta: Array) -> Array:
    """``R`` with rows ``(cos, sin)`` / ``(-sin, cos)``: local = R @ (x - x_i).

    With theta the normal's angle, the first local axis points along the outward
    normal, so local +x is the outward direction.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.empty((len(theta), 2, 2))
    R[:, 0, 0] = c
    R[:, 0, 1] = s
    R[:, 1, 0] = -s
    R[:, 1, 1] = c
    return R


def rotate_offsets(offsets: Array, theta: Array) -> Array:
    """Rotate per-stencil neighbour offsets (S, K, 2) into the canonical frame."""
    R = rotation_matrices(theta)
    return np.einsum("sij,skj->ski", R, offsets)


def weights_to_global(w_local_x: Array, w_local_y: Array, theta: Array
                      ) -> tuple[Array, Array]:
    """Map first-derivative weights out of the canonical frame.

    ``w_local_x`` differentiates along the first local axis, which is the global
    direction ``(cos t, sin t)``; ``w_local_y`` along ``(-sin t, cos t)``.  Hence

        w_x_global = cos t * w'_x - sin t * w'_y
        w_y_global = sin t * w'_x + cos t * w'_y
    """
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    return c * w_local_x - s * w_local_y, s * w_local_x + c * w_local_y


def directional_weights(w_x: Array, w_y: Array, direction: Array) -> Array:
    """Weights for the derivative along an arbitrary unit vector, in any one frame.

    Neumann rows use ``direction`` = outward normal; Robin rows add ``a * I`` to the
    assembled row.  Note that in the canonical frame this is just ``w_x``.
    """
    d = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-14)
    return d[:, 0:1] * w_x + d[:, 1:2] * w_y
