"""Diagnostic figures.

Two things are worth looking at before trusting any numbers.

``plot_cloud``
    Does the node set look right -- is the wall layer where you think it is, and do
    the outward normals point out?  Worth running on your own mesh first: a flipped
    normal or a mislabelled interior node turns every boundary row into nonsense, and
    it is obvious in a picture and invisible in a loss curve.

``plot_error_field``
    *Where* the operator is weak, rather than the single aggregate number the
    evaluation prints.  Error concentrated at a re-entrant corner means something
    different from error spread evenly across the interior.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from functions.domains import NodeCloud


def plot_cloud(cloud: NodeCloud, path: Optional[str] = None, normals: bool = True,
               ax=None, title: Optional[str] = None):
    """Scatter the cloud, highlighting boundary nodes and their outward normals."""
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6, 6))

    interior = ~cloud.is_boundary
    ax.scatter(cloud.coords[interior, 0], cloud.coords[interior, 1],
               s=4, c="0.65", linewidths=0, label="interior")
    ax.scatter(cloud.coords[~interior, 0], cloud.coords[~interior, 1],
               s=12, c="crimson", linewidths=0, label="boundary")

    if normals and cloud.is_boundary.any():
        b = cloud.coords[cloud.is_boundary]
        n = cloud.normals[cloud.is_boundary]
        ax.quiver(b[:, 0], b[:, 1], n[:, 0], n[:, 1],
                  color="navy", scale=22, width=0.004, label="outward normal")

    ax.set_aspect("equal")
    ax.set_title(title or f"{cloud.name}  N={cloud.n_nodes}  "
                          f"boundary={int(cloud.is_boundary.sum())}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if created:
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        if path:
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(path, dpi=110)
            plt.close()
    return ax


def plot_error_field(cloud: NodeCloud,
                     error: NDArray[np.float64],
                     path: Optional[str] = None,
                     title: str = "operator error",
                     log_scale: bool = True):
    """Colour each node by its pointwise operator error."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    e = np.abs(np.asarray(error, dtype=float))
    fig, ax = plt.subplots(figsize=(6.5, 6))

    norm = None
    if log_scale:
        positive = e[e > 0]
        if positive.size:
            norm = LogNorm(vmin=max(positive.min(), positive.max() * 1e-6),
                           vmax=positive.max())

    sc = ax.scatter(cloud.coords[:, 0], cloud.coords[:, 1], c=np.maximum(e, 1e-300),
                    s=9, cmap="viridis", norm=norm, linewidths=0)
    # ring the wall nodes so it is clear whether the error sits on the boundary
    if cloud.is_boundary.any():
        b = cloud.coords[cloud.is_boundary]
        ax.scatter(b[:, 0], b[:, 1], s=26, facecolors="none", edgecolors="crimson",
                   linewidths=0.6, label="boundary")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.colorbar(sc, ax=ax, label="|error|")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=110)
        plt.close()
    return ax
