"""Stage 3 -- evaluate the boundary-aware operator on a held-out domain.

Reports, split into interior and wall nodes:

* moment (Taylor consistency) residuals, against the classical weighted least-squares
  solve on the identical stencils;
* errors in d/dx, d/dy and the Laplacian of an analytic field;
* the normal-derivative error at boundary nodes -- the quantity a Neumann or Robin
  row depends on;
* the largest weight magnitude, which is how badly conditioned one-sided stencils
  show up in practice.

The holdout shape is not in the training mix.

Run:
    python main_eval_bc.py
"""

from __future__ import annotations

import logging
from os.path import join as jn

import numpy as np

from functions.boundary import weights_to_global
from functions.classical import (classical_weights_batch, moment_residual_batch,
                                 stencil_conditioning)
from functions.domains import holdout_domain
from functions.infer import predict_weights
from functions.stencils import extract_stencils
from models.nemdo_bc import load_bc_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Analytic test field and its derivatives.
def u_fn(x, y):   return np.sin(3 * x) * np.cos(2 * y) + x ** 3
def ux_fn(x, y):  return 3 * np.cos(3 * x) * np.cos(2 * y) + 3 * x ** 2
def uy_fn(x, y):  return -2 * np.sin(3 * x) * np.sin(2 * y)
def lap_fn(x, y): return -13 * np.sin(3 * x) * np.cos(2 * y) + 6 * x


def _rel_l2(pred: np.ndarray, exact: np.ndarray) -> float:
    denom = np.linalg.norm(exact)
    return float(np.linalg.norm(pred - exact) / denom) if denom > 0 else float("nan")


def _apply(w: np.ndarray, u_stencil: np.ndarray) -> np.ndarray:
    """sum_j (u_j - u_i) w_ji."""
    return np.sum((u_stencil - u_stencil[:, 0:1]) * w, axis=1)


def evaluate(model_path: str, spacing: float, stencil_size: int,
             order: int = 2, kernel: str = "wendland",
             figure_dir: str | None = None) -> None:
    model, attrs = load_bc_model(model_path)
    ops = list(model.operators)
    logger.info("Model %s | operators %s | trained %d epochs | best val %.3e",
                model_path, ops, attrs.get("epochs", -1), attrs.get("best_val_loss", np.nan))

    cloud = holdout_domain(spacing=spacing)
    stencils = extract_stencils(cloud, stencil_size=stencil_size, cloud_index=-1)
    logger.info("Holdout '%s': %d nodes, %d on the boundary",
                cloud.name, cloud.n_nodes, int(cloud.is_boundary.sum()))

    nb_idx = stencils.neighbour_index     # carried with the stencils, so the ordering
                                          # cannot drift out of sync with the weights
    pred = predict_weights(model, stencils)
    offs_local = stencils.offsets_local                       # canonical frame, physical
    classical = classical_weights_batch(offs_local, ops, order=order, kernel=kernel)

    wall = stencils.is_boundary
    inter = ~wall
    xs, ys = cloud.coords[:, 0], cloud.coords[:, 1]
    u_st = u_fn(cloud.coords[nb_idx][:, :, 0], cloud.coords[nb_idx][:, :, 1])

    # ---------------- moment residuals ----------------
    logger.info("")
    logger.info("Moment residual RMS (Taylor consistency; 0 = exact)")
    logger.info("  %-10s %12s %12s %12s %12s", "operator", "NeMDO int", "NeMDO wall",
                "classic int", "classic wall")
    res_n = moment_residual_batch(pred["local"].astype(np.float64), offs_local, ops, order)
    res_c = moment_residual_batch(classical, offs_local, ops, order)
    rms_n = np.sqrt((res_n ** 2).mean(axis=1))            # (S, H)
    rms_c = np.sqrt((res_c ** 2).mean(axis=1))
    for c, op in enumerate(ops):
        logger.info("  %-10s %12.3e %12.3e %12.3e %12.3e", op,
                    rms_n[inter, c].mean(), rms_n[wall, c].mean(),
                    rms_c[inter, c].mean(), rms_c[wall, c].mean())

    # ---------------- derivative errors on an analytic field ----------------
    exact = {"x": ux_fn(xs, ys), "y": uy_fn(xs, ys), "laplace": lap_fn(xs, ys)}
    logger.info("")
    logger.info("Relative L2 error on u = sin(3x)cos(2y) + x^3")
    logger.info("  %-10s %12s %12s %12s %12s", "operator", "NeMDO int", "NeMDO wall",
                "classic int", "classic wall")
    for c, op in enumerate(ops):
        if op not in exact:
            continue
        pn = _apply(pred["global"][:, :, c], u_st)
        # classical weights live in the canonical frame too; rotate the first
        # derivatives back so both are compared on global axes
        pc_w = classical[:, :, c]
        if op in ("x", "y") and "x" in ops and "y" in ops:
            gx, gy = weights_to_global(classical[:, :, ops.index("x")],
                                       classical[:, :, ops.index("y")],
                                       stencils.theta.astype(np.float64))
            pc_w = gx if op == "x" else gy
        pc = _apply(pc_w, u_st)
        logger.info("  %-10s %12.3e %12.3e %12.3e %12.3e", op,
                    _rel_l2(pn[inter], exact[op][inter]), _rel_l2(pn[wall], exact[op][wall]),
                    _rel_l2(pc[inter], exact[op][inter]), _rel_l2(pc[wall], exact[op][wall]))

    # ---------------- Neumann: normal derivative at the wall ----------------
    if pred["normal"] is not None:
        n = cloud.normals[wall]
        dudn_exact = n[:, 0] * ux_fn(xs[wall], ys[wall]) + n[:, 1] * uy_fn(xs[wall], ys[wall])
        dudn_nemdo = _apply(pred["normal"][wall], u_st[wall])
        dudn_class = _apply(classical[wall][:, :, ops.index("x")], u_st[wall])
        logger.info("")
        logger.info("Normal derivative at %d boundary nodes (the Neumann/Robin row)", wall.sum())
        logger.info("  NeMDO     rel L2 %.3e   max abs err %.3e",
                    _rel_l2(dudn_nemdo, dudn_exact), np.max(np.abs(dudn_nemdo - dudn_exact)))
        logger.info("  classical rel L2 %.3e   max abs err %.3e",
                    _rel_l2(dudn_class, dudn_exact), np.max(np.abs(dudn_class - dudn_exact)))

    # ---------------- conditioning and weight magnitudes ----------------
    rng = np.random.default_rng(0)
    si = rng.choice(np.flatnonzero(inter), size=min(200, int(inter.sum())), replace=False)
    sw = rng.choice(np.flatnonzero(wall), size=min(200, int(wall.sum())), replace=False)
    cond_i = np.median([stencil_conditioning(offs_local[i], order) for i in si])
    cond_w = np.median([stencil_conditioning(offs_local[i], order) for i in sw])
    lap_c = ops.index("laplace") if "laplace" in ops else 0
    logger.info("")
    logger.info("Stencil difficulty and weight magnitude (%s head)", ops[lap_c])
    logger.info("  median condition number : interior %.3g, wall %.3g (%.1fx harder)",
                cond_i, cond_w, cond_w / cond_i)
    logger.info("  max |w| NeMDO           : interior %.3g, wall %.3g",
                np.abs(pred["local"][inter, :, lap_c]).max(),
                np.abs(pred["local"][wall, :, lap_c]).max())
    logger.info("  max |w| classical       : interior %.3g, wall %.3g",
                np.abs(classical[inter, :, lap_c]).max(),
                np.abs(classical[wall, :, lap_c]).max())

    if figure_dir:
        import os
        from functions.plots import plot_cloud, plot_error_field
        os.makedirs(figure_dir, exist_ok=True)
        plot_cloud(cloud, path=jn(figure_dir, "holdout_cloud.png"))
        lap_err = _apply(pred["global"][:, :, lap_c], u_st) - lap_fn(xs, ys)
        plot_error_field(cloud, lap_err, path=jn(figure_dir, "laplace_error.png"),
                         title="NeMDO Laplacian error on the holdout shape")
        logger.info("")
        logger.info("Figures written to %s", figure_dir)


if __name__ == "__main__":
    model_path   = jn("saved_models_bc", "nemdo_bc_1.pth")
    spacing      = 0.022 * 0.9      # matches main_build_dataset.py's holdout spacing
    stencil_size = 35
    order        = 2
    figure_dir   = "plots"          # set to None to skip the diagnostic figures

    evaluate(model_path=model_path, spacing=spacing, stencil_size=stencil_size,
             order=order, figure_dir=figure_dir)
