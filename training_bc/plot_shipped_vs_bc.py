"""Figure for main_eval_shipped.py: Laplacian error per node and versus wall distance.

Run after ``python main_eval_shipped.py`` (from ``training_bc/``):
    python plot_shipped_vs_bc.py
"""

from __future__ import annotations

from os.path import join as jn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from functions.domains import holdout_domain

LABELS = {"shipped_nemdo_x_lap": "shipped nemdo_x / nemdo_lap",
          "nemdo_bc": "nemdo_bc (boundary-aware)",
          "classical_WLS": "classical WLS"}
COLORS = {"shipped_nemdo_x_lap": "#c2410c", "nemdo_bc": "#2563eb", "classical_WLS": "#525252"}


def main(results_dir: str = "results", spacing: float = 0.022 * 0.9) -> None:
    d = np.load(jn(results_dir, "laplace_error_fields.npz"))
    coords = d["coords"]
    keys = [k for k in LABELS if k in d.files]
    errs = {k: np.abs(d[k]) for k in keys}
    dist = holdout_domain(spacing=spacing).boundary_distance()
    h = np.median(np.sort(np.linalg.norm(coords[:, None] - coords[None], axis=2), axis=1)[:, 1])

    allv = np.concatenate([e for e in errs.values()])
    norm = LogNorm(vmin=max(np.percentile(allv, 2), 1e-4), vmax=np.percentile(allv, 99.5))

    fig = plt.figure(figsize=(4.2 * len(keys) + 4.8, 4.4))
    gs = fig.add_gridspec(1, len(keys) + 1, width_ratios=[1] * len(keys) + [1.25], wspace=0.25)
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=errs[k], s=7, cmap="magma_r", norm=norm)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(LABELS[k], fontsize=10)
    fig.colorbar(sc, ax=fig.axes[:len(keys)], shrink=0.8, label="|Laplacian error|  (exact |Δu| ≈ 1–15)")

    ax = fig.add_subplot(gs[0, -1])
    layer = np.round(dist / h).astype(int)
    for k in keys:
        lv = np.arange(0, min(layer.max(), 8) + 1)
        med = [np.median(errs[k][layer == l]) for l in lv]
        ax.semilogy(lv, med, "o-", color=COLORS[k], label=LABELS[k], ms=4)
    ax.set_xlabel("distance from wall (node layers)")
    ax.set_ylabel("median |Laplacian error|")
    ax.set_title("Error vs distance from the wall", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False)

    out = jn(results_dir, "shipped_vs_bc_laplacian.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
