"""Read an external node set so the operator can be trained on, or applied to, a real mesh.

The generated domains in ``domains.py`` are for building a general-purpose operator.
To specialise to the geometry you actually solve on -- a COMSOL mesh, say -- point
this at the node coordinates and feed the result to ``stencils.extract_stencils``.
Still self-supervised: only node positions and which of them lie on a wall are used,
never the solution field.

CSV layout (header optional, comma separated)::

    x, y[, is_boundary[, nx, ny]]

``is_boundary`` is 0/1.  If the normals are absent they are estimated from the point
distribution; if ``is_boundary`` is absent too, both are estimated.  Supply the real
values when the mesh knows them -- estimated normals degrade at sharp corners, which
is exactly where the stencils are hardest.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from functions.boundary import estimate_boundary_and_normals
from functions.domains import NodeCloud

logger = logging.getLogger(__name__)


def load_csv_cloud(path: str, name: str | None = None,
                   estimate_k: int = 12) -> NodeCloud:
    """Read ``x, y[, is_boundary[, nx, ny]]`` into a :class:`NodeCloud`."""
    raw = np.genfromtxt(path, delimiter=",", dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    if np.isnan(raw[0]).any():          # header row
        raw = raw[1:]
    if raw.shape[1] < 2:
        raise ValueError(f"{path}: expected at least 2 columns (x, y), got {raw.shape[1]}")

    coords = raw[:, :2]
    is_boundary = raw[:, 2].astype(bool) if raw.shape[1] >= 3 else None
    normals = raw[:, 3:5] if raw.shape[1] >= 5 else None

    return build_cloud(coords, is_boundary=is_boundary, normals=normals,
                       name=name or path, estimate_k=estimate_k)


def build_cloud(coords: NDArray[np.float64],
                is_boundary: NDArray[np.bool_] | None = None,
                normals: NDArray[np.float64] | None = None,
                name: str = "mesh",
                estimate_k: int = 12) -> NodeCloud:
    """Wrap raw coordinates as a :class:`NodeCloud`, filling in what is missing."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {coords.shape}")

    if is_boundary is None or normals is None:
        est_b, est_n, _ = estimate_boundary_and_normals(coords, k=estimate_k)
        if is_boundary is None:
            is_boundary = est_b
            logger.info("%s: estimated %d boundary nodes of %d from point asymmetry",
                        name, int(est_b.sum()), len(coords))
        if normals is None:
            normals = est_n
            logger.info("%s: estimated outward normals from point asymmetry "
                        "(supply real ones if the mesh has them)", name)

    normals = np.asarray(normals, dtype=float).copy()
    is_boundary = np.asarray(is_boundary, dtype=bool)
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(nrm < 1e-14, 1.0, nrm))
    normals[~is_boundary] = 0.0

    spacing = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    return NodeCloud(coords=coords, is_boundary=is_boundary, normals=normals,
                     spacing=spacing, name=name)


def save_csv_cloud(path: str, cloud: NodeCloud) -> None:
    """Round-trip a cloud back out, e.g. to inspect estimated normals."""
    arr = np.column_stack([cloud.coords, cloud.is_boundary.astype(int), cloud.normals])
    np.savetxt(path, arr, delimiter=",", header="x,y,is_boundary,nx,ny", comments="")
