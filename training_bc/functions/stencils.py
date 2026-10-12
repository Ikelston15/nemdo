"""Stencil extraction and batching for boundary-aware training.

Keeps the conventions of ``training/``: a stencil is the ``K`` nearest nodes with the
centre first and the rest sorted by distance, offsets are normalised by the farthest
neighbour distance ``h``, and the graph is a star (centre <-> each neighbour, both
directions).  Because the centre sits at offset ``(0, 0)`` it contributes nothing to
any moment of order >= 1, which is what makes the operator form

    L u(x_i) = sum_j (u_j - u_i) w_ji

consistent: the centre weight is implied, ``w_ii = -sum_j w_ji``.

Two things are added on top of the original pipeline.  Offsets are rotated into the
canonical normal-aligned frame (see ``boundary.py``), and each node carries a
boundary-proximity channel.  The feature layout is

    [ dx/h, dy/h, is_centre, boundary_proximity ]

with ``dx/h, dy/h`` kept in columns 0 and 1 so the repository's moment code can be
reused unchanged.

Since ``K`` is fixed, batching is done with a precomputed edge-index template rather
than PyG's collation -- the same trick ``testing/functions/nemdo_operator.py`` uses,
and far faster than building ``Data`` objects for every stencil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from torch import Tensor

from functions.boundary import (boundary_proximity, canonical_angle,
                                estimate_boundary_and_normals, rotate_offsets)
from functions.domains import NodeCloud

Array = NDArray[np.float64]

N_FEATURES = 4


@dataclass
class StencilSet:
    """A stack of stencils, ready to be fed to the model."""

    features: NDArray[np.float32]     # (S, K, 4)
    h: NDArray[np.float32]            # (S,) stencil radius
    theta: NDArray[np.float32]        # (S,) canonical rotation applied
    proximity: NDArray[np.float32]    # (S,) 1 on the wall, 0 deep inside
    is_boundary: NDArray[np.bool_]    # (S,) centre node lies on the boundary
    node_index: NDArray[np.int64]     # (S,) index into the source cloud
    cloud_index: NDArray[np.int64]    # (S,) which cloud it came from
    normals: NDArray[np.float32]      # (S, 2) outward normal in the global frame
    neighbour_index: NDArray[np.int64]  # (S, K) indices into the source cloud, centre first

    def __len__(self) -> int:
        return len(self.h)

    @property
    def stencil_size(self) -> int:
        return self.features.shape[1]

    @property
    def offsets_local(self) -> Array:
        """Physical (un-normalised) offsets in the canonical frame."""
        return self.features[:, :, :2].astype(np.float64) * self.h[:, None, None]

    def subset(self, idx: NDArray[np.int64]) -> "StencilSet":
        return StencilSet(features=self.features[idx], h=self.h[idx],
                          theta=self.theta[idx], proximity=self.proximity[idx],
                          is_boundary=self.is_boundary[idx],
                          node_index=self.node_index[idx],
                          cloud_index=self.cloud_index[idx],
                          normals=self.normals[idx],
                          neighbour_index=self.neighbour_index[idx])

    def save(self, path: str) -> None:
        np.savez_compressed(path, features=self.features, h=self.h, theta=self.theta,
                            proximity=self.proximity, is_boundary=self.is_boundary,
                            node_index=self.node_index, cloud_index=self.cloud_index,
                            normals=self.normals, neighbour_index=self.neighbour_index)

    @staticmethod
    def load(path: str) -> "StencilSet":
        d = np.load(path)
        nb = (d["neighbour_index"] if "neighbour_index" in d.files
              else np.full(d["features"].shape[:2], -1, dtype=np.int64))
        return StencilSet(features=d["features"], h=d["h"], theta=d["theta"],
                          proximity=d["proximity"], is_boundary=d["is_boundary"],
                          node_index=d["node_index"], cloud_index=d["cloud_index"],
                          normals=d["normals"], neighbour_index=nb)


def concat(sets: Sequence[StencilSet]) -> StencilSet:
    return StencilSet(
        features=np.concatenate([s.features for s in sets]),
        h=np.concatenate([s.h for s in sets]),
        theta=np.concatenate([s.theta for s in sets]),
        proximity=np.concatenate([s.proximity for s in sets]),
        is_boundary=np.concatenate([s.is_boundary for s in sets]),
        node_index=np.concatenate([s.node_index for s in sets]),
        cloud_index=np.concatenate([s.cloud_index for s in sets]),
        normals=np.concatenate([s.normals for s in sets]),
        neighbour_index=np.concatenate([s.neighbour_index for s in sets]),
    )


def extract_stencils(cloud: NodeCloud,
                     stencil_size: int = 35,
                     cloud_index: int = 0,
                     canonical_frame: bool = True) -> StencilSet:
    """Build one stencil per node of ``cloud``."""
    coords = cloud.coords
    k = min(stencil_size, len(coords))
    if k < stencil_size:
        raise ValueError(f"cloud '{cloud.name}' has {len(coords)} nodes, "
                         f"fewer than stencil_size={stencil_size}")

    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k)               # already sorted, self first
    offsets = coords[idx] - coords[:, None, :]        # (N, K, 2)
    h = dist[:, -1]
    h = np.where(h < 1e-14, 1e-14, h)

    d_bnd = cloud.boundary_distance()
    prox = boundary_proximity(d_bnd, h)
    normals = cloud.nearest_boundary_normal()

    if canonical_frame:
        theta = canonical_angle(normals, prox)
        offsets = rotate_offsets(offsets, theta)
    else:
        theta = np.zeros(len(coords))

    feats = np.zeros((len(coords), k, N_FEATURES), dtype=np.float32)
    feats[:, :, :2] = (offsets / h[:, None, None]).astype(np.float32)
    feats[:, 0, 2] = 1.0                               # is_centre
    feats[:, :, 3] = prox[:, None].astype(np.float32)

    return StencilSet(features=feats,
                      h=h.astype(np.float32),
                      theta=theta.astype(np.float32),
                      proximity=prox.astype(np.float32),
                      is_boundary=cloud.is_boundary.copy(),
                      node_index=np.arange(len(coords), dtype=np.int64),
                      cloud_index=np.full(len(coords), cloud_index, dtype=np.int64),
                      normals=normals.astype(np.float32),
                      neighbour_index=idx.astype(np.int64))


def stencils_from_coords(coords: Array,
                         stencil_size: int = 35,
                         is_boundary: NDArray[np.bool_] | None = None,
                         normals: Array | None = None,
                         canonical_frame: bool = True) -> StencilSet:
    """Stencils for an arbitrary point cloud, e.g. nodes read out of a mesh.

    If ``is_boundary``/``normals`` are not supplied they are estimated from the point
    distribution.  Supply them when the mesh knows them -- an estimate near a sharp
    corner is noticeably worse than the real thing.
    """
    coords = np.asarray(coords, dtype=float)
    if is_boundary is None or normals is None:
        est_b, est_n, _ = estimate_boundary_and_normals(coords)
        is_boundary = est_b if is_boundary is None else is_boundary
        normals = est_n if normals is None else normals

    spacing = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    cloud = NodeCloud(coords=coords, is_boundary=np.asarray(is_boundary, dtype=bool),
                      normals=np.asarray(normals, dtype=float), spacing=spacing,
                      name="external")
    return extract_stencils(cloud, stencil_size=stencil_size,
                            canonical_frame=canonical_frame)


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------
def star_edge_index(stencil_size: int, batch_size: int) -> Tensor:
    """Edge index for ``batch_size`` disjoint star graphs of ``stencil_size`` nodes."""
    nb = torch.arange(1, stencil_size, dtype=torch.long)
    base = torch.cat([torch.stack([nb, torch.zeros_like(nb)]),
                      torch.stack([torch.zeros_like(nb), nb])], dim=1)   # (2, 2(K-1))
    offs = (torch.arange(batch_size, dtype=torch.long) * stencil_size)
    return (base[:, None, :] + offs[None, :, None]).reshape(2, -1)


class StencilBatcher:
    """Iterates fixed-size batches of stencils as flat (x, edge_index, batch) tensors."""

    def __init__(self, data: StencilSet, batch_size: int, shuffle: bool = False,
                 drop_last: bool = False, device: str = "cpu", seed: int = 0):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.k = data.stencil_size
        self._x = torch.from_numpy(data.features)                     # (S, K, F)
        self._edge_cache: dict[int, Tensor] = {}
        self._batch_cache: dict[int, Tensor] = {}

    def __len__(self) -> int:
        n = len(self.data)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def _edges(self, b: int) -> Tensor:
        if b not in self._edge_cache:
            self._edge_cache[b] = star_edge_index(self.k, b).to(self.device)
        return self._edge_cache[b]

    def _batch_vec(self, b: int) -> Tensor:
        if b not in self._batch_cache:
            self._batch_cache[b] = torch.arange(
                b, dtype=torch.long, device=self.device).repeat_interleave(self.k)
        return self._batch_cache[b]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, NDArray[np.int64]]]:
        order = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            sel = order[start:start + self.batch_size]
            if self.drop_last and len(sel) < self.batch_size:
                break
            b = len(sel)
            x = self._x[sel].reshape(b * self.k, -1).to(self.device, non_blocking=True)
            yield x, self._edges(b), self._batch_vec(b), sel
