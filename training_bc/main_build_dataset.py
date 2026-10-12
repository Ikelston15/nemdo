"""Stage 1 -- build a stencil dataset from domains that have boundaries.

Mirrors ``training/main_import.py``: generate the data, split it, write it to disk
for the training script to pick up.  The difference is where the data comes from.
Instead of reading pre-computed interior stencils out of ``ij_link*.csv``, this
generates node clouds in bounded domains (square, disk, annulus, L-shape, star
blobs), so the set contains the one-sided stencils that occur at a wall.

Run:
    python main_build_dataset.py
"""

from __future__ import annotations

import logging
import os
from os.path import join as jn

import numpy as np

from functions.classical import stencil_conditioning
from functions.domains import default_domain_mix, holdout_domain
from functions.stencils import concat, extract_stencils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build(spacing: float,
          stencil_size: int,
          seed: int,
          out_dir: str,
          val_fraction: float = 0.15,
          boundary_oversample: int = 2) -> None:
    os.makedirs(out_dir, exist_ok=True)

    clouds = default_domain_mix(spacing=spacing, seed=seed)
    sets = []
    for i, cloud in enumerate(clouds):
        s = extract_stencils(cloud, stencil_size=stencil_size, cloud_index=i)
        n_b = int(cloud.is_boundary.sum())
        logger.info("%-16s %5d nodes (%4d on the boundary), spacing %.4f",
                    cloud.name, cloud.n_nodes, n_b, cloud.spacing)
        sets.append(s)
    data = concat(sets)

    # Boundary stencils are a minority by construction (O(sqrt(N)) of N nodes).
    # Repeating them costs nothing in memory terms and keeps the gradient signal
    # from being swamped by easy interior stencils. The loss also reweights by
    # proximity; this just makes sure the hard cases are seen often enough.
    if boundary_oversample > 1:
        # Only the genuinely one-sided ones. "proximity > 0" reaches a full stencil
        # radius inside, which is most of a domain at K = 35 and would tip the whole
        # set toward the wall; the loss already reweights the mildly affected rows.
        extra_idx = np.flatnonzero(data.proximity > 0.5)
        reps = np.tile(extra_idx, boundary_oversample - 1)
        data = concat([data, data.subset(reps)])
        logger.info("Oversampled %d strongly one-sided stencils x%d",
                    len(extra_idx), boundary_oversample)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    n_val = int(val_fraction * len(data))
    val_idx, train_idx = order[:n_val], order[n_val:]

    train, val = data.subset(train_idx), data.subset(val_idx)
    train.save(jn(out_dir, "train.npz"))
    val.save(jn(out_dir, "val.npz"))

    hold = extract_stencils(holdout_domain(spacing=spacing * 0.9), stencil_size=stencil_size,
                            cloud_index=-1)
    hold.save(jn(out_dir, "holdout.npz"))
    np.save(jn(out_dir, "holdout_coords.npy"), holdout_domain(spacing=spacing * 0.9).coords)

    # A quick look at how much harder the boundary stencils actually are.
    sample = rng.choice(len(data), size=min(400, len(data)), replace=False)
    sub = data.subset(sample)
    cond = np.array([stencil_conditioning(o) for o in sub.offsets_local])
    interior, wall = cond[sub.proximity == 0.0], cond[sub.is_boundary]
    logger.info("Stencils: %d train, %d val, %d holdout (K=%d)",
                len(train), len(val), len(hold), stencil_size)
    logger.info("Train mix: %.1f%% on a wall, %.1f%% within one stencil radius of one",
                100 * train.is_boundary.mean(), 100 * (train.proximity > 0).mean())
    if len(interior) and len(wall):
        logger.info("Median moment-matrix condition number: interior %.3g, wall %.3g (%.1fx)",
                    np.median(interior), np.median(wall),
                    np.median(wall) / np.median(interior))
    logger.info("Wrote dataset to %s", out_dir)


if __name__ == "__main__":
    spacing            = 0.022      # nominal node spacing; smaller = more stencils
    stencil_size       = 35         # K, matches the shipped nemdo_x / nemdo_lap models
    seed               = 1
    boundary_oversample = 2         # repeat near-boundary stencils this many times
    out_dir            = "preproc_data_bc"

    build(spacing=spacing, stencil_size=stencil_size, seed=seed,
          out_dir=out_dir, boundary_oversample=boundary_oversample)
