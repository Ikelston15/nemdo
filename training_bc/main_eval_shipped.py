"""Run the shipped interior-only checkpoints (``nemdo_x``, ``nemdo_lap``) on boundary stencils.

The shipped models were trained on fully surrounded stencils from a perturbed square
grid.  This script feeds them the held-out boundary domain used by
``main_eval_bc.py`` and reports how they do, side by side with the boundary-aware
``nemdo_bc`` model (if a checkpoint exists) and the classical weighted least-squares
solve -- all on identical K = 35 neighbour sets.

How each operator is obtained from the shipped models
-----------------------------------------------------
* ``d/dx``      : ``nemdo_x`` on global-frame offsets ``(dx, dy)/h`` -- exactly as shipped.
* ``d/dy``      : ``nemdo_x`` on offsets rotated by -90 deg, ``(dy, -dx)/h``; the derivative
                  along the new x axis is the derivative along global y.
* ``Laplacian`` : ``nemdo_lap`` on global-frame offsets (rotation invariant).
* ``d/dn``      : ``nemdo_x`` on offsets in the canonical frame (outward normal on +x),
                  i.e. the same frame ``nemdo_bc`` uses.  This is the most favourable way
                  to get a Neumann row out of the shipped model.

Nodes are split three ways:
* deep interior : stencil does not reach a wall (proximity 0) -- in-distribution for nemdo_x;
* near wall     : stencil reaches a wall but the centre is not on it (partly one-sided);
* wall          : centre on the boundary (fully one-sided).

Run (from ``training_bc/``):
    python main_eval_shipped.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from os.path import join as jn

import numpy as np
import torch

from functions.boundary import rotate_offsets
from functions.classical import classical_weights_batch, moment_residual_batch
from functions.domains import holdout_domain
from functions.stencils import extract_stencils

HERE = os.path.dirname(os.path.abspath(__file__))
TESTING = os.path.normpath(jn(HERE, "..", "testing"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
for noisy in ("models.preproc", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ------------------------------------------------------------------ test fields
def _field_trig():
    u = lambda x, y: np.sin(3 * x) * np.cos(2 * y) + x ** 3
    ux = lambda x, y: 3 * np.cos(3 * x) * np.cos(2 * y) + 3 * x ** 2
    uy = lambda x, y: -2 * np.sin(3 * x) * np.sin(2 * y)
    lap = lambda x, y: -13 * np.sin(3 * x) * np.cos(2 * y) + 6 * x
    return "sin(3x)cos(2y)+x^3", u, ux, uy, lap


def _field_gauss(a: float = 0.08, x0: float = 0.15, y0: float = -0.1):
    g = lambda x, y: np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / a)
    ux = lambda x, y: -2 * (x - x0) / a * g(x, y)
    uy = lambda x, y: -2 * (y - y0) / a * g(x, y)
    lap = lambda x, y: g(x, y) * (4 * ((x - x0) ** 2 + (y - y0) ** 2) / a ** 2 - 4 / a)
    return "off-centre Gaussian", g, ux, uy, lap


FIELDS = [_field_trig(), _field_gauss()]


# ------------------------------------------------------------------ shipped models
def load_shipped(name: str):
    """Load a shipped NEMDO_X_LAP checkpoint without going through testing/'s
    ``load_gnn`` (which also needs the optimizer state and logs noisily)."""
    sys.path.insert(0, TESTING)
    try:
        from models.nemdo_x_and_lap import NEMDO_X_LAP
    finally:
        sys.path.pop(0)
    attrs = torch.load(jn(TESTING, "models", "trained_model", f"{name}.pth"),
                       map_location="cpu", weights_only=False)
    model = NEMDO_X_LAP(embedding_size=attrs["embedding_size"], layers=attrs["layers"])
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in attrs["weights"].items()}
    model.load_state_dict(state)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    return model, n_par


@torch.no_grad()
def run_shipped(model, offsets_norm: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Star-graph inference on (S, K, 2) normalised offsets -> (S, K) normalised weights."""
    S, K, _ = offsets_norm.shape
    out = np.empty((S, K), dtype=np.float64)
    leaves = torch.arange(1, K)
    one = torch.stack([torch.cat([leaves, torch.zeros(K - 1, dtype=torch.long)]),
                       torch.cat([torch.zeros(K - 1, dtype=torch.long), leaves])])
    for s0 in range(0, S, batch_size):
        s1 = min(S, s0 + batch_size)
        b = s1 - s0
        x = torch.from_numpy(offsets_norm[s0:s1].reshape(-1, 2).astype(np.float32))
        shift = (torch.arange(b) * K).repeat_interleave(one.shape[1])
        edge_index = one.repeat(1, b) + shift[None, :]
        batch = torch.arange(b).repeat_interleave(K)
        out[s0:s1] = model(x, edge_index, batch).view(b, K).numpy()
    return out


# ------------------------------------------------------------------ helpers
def rel_l2(pred, exact):
    d = np.linalg.norm(exact)
    return float(np.linalg.norm(pred - exact) / d) if d > 0 else float("nan")


def apply(w, u_st):
    return np.sum((u_st - u_st[:, 0:1]) * w, axis=1)


def evaluate(spacing: float = 0.022 * 0.9, stencil_size: int = 35, order: int = 2,
             bc_model_path: str | None = jn("saved_models_bc", "nemdo_bc_1.pth"),
             out_json: str | None = jn("results", "shipped_on_boundary.json")) -> dict:
    cloud = holdout_domain(spacing=spacing)
    st = extract_stencils(cloud, stencil_size=stencil_size, cloud_index=-1)  # canonical frame
    nb = st.neighbour_index
    h = st.h.astype(np.float64)
    off_glob = cloud.coords[nb] - cloud.coords[:, None, :]                   # (S, K, 2)
    off_can = st.offsets_local                                               # canonical frame
    theta = st.theta.astype(np.float64)

    wall = st.is_boundary
    near = (st.proximity > 0) & ~wall
    deep = st.proximity == 0
    regions = {"deep interior": deep, "near wall": near, "wall": wall}
    logger.info("Holdout '%s': %d nodes -> deep interior %d, near wall %d, wall %d",
                cloud.name, cloud.n_nodes, deep.sum(), near.sum(), wall.sum())

    # ---------------- weights from every method, global frame, physical units -------
    methods: dict[str, dict[str, np.ndarray]] = {}

    mx, npx = load_shipped("nemdo_x")
    ml, npl = load_shipped("nemdo_lap")
    logger.info("Shipped nemdo_x %d params, nemdo_lap %d params", npx, npl)
    g_n = off_glob / h[:, None, None]
    rot90 = np.stack([g_n[..., 1], -g_n[..., 0]], axis=-1)       # x' = y, y' = -x
    w_sx = run_shipped(mx, g_n) / h[:, None]
    w_sy = run_shipped(mx, rot90) / h[:, None]
    w_sl = run_shipped(ml, g_n) / h[:, None] ** 2
    w_sn = run_shipped(mx, off_can / h[:, None, None]) / h[:, None]          # d/dn at wall
    methods["shipped nemdo_x/lap"] = {"x": w_sx, "y": w_sy, "laplace": w_sl, "normal": w_sn}

    # classical WLS, built in the canonical frame and rotated back (as main_eval_bc does)
    ops = ["x", "y", "laplace"]
    cw = classical_weights_batch(off_can, ops, order=order)
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    methods["classical WLS"] = {"x": c * cw[..., 0] - s * cw[..., 1],
                                "y": s * cw[..., 0] + c * cw[..., 1],
                                "laplace": cw[..., 2], "normal": cw[..., 0]}

    if bc_model_path and os.path.exists(bc_model_path):
        from functions.infer import predict_weights
        from models.nemdo_bc import load_bc_model
        bc, attrs = load_bc_model(bc_model_path)
        p = predict_weights(bc, st)
        o = list(bc.operators)
        methods["nemdo_bc"] = {"x": p["global"][..., o.index("x")],
                               "y": p["global"][..., o.index("y")],
                               "laplace": p["global"][..., o.index("laplace")],
                               "normal": p["normal"]}
        logger.info("nemdo_bc: %s (%d epochs, best val %.3e)", bc_model_path,
                    attrs.get("epochs", -1), attrs.get("best_val_loss", np.nan))
    else:
        logger.info("nemdo_bc checkpoint not found at %s -- skipped", bc_model_path)

    results: dict = {"n_nodes": int(cloud.n_nodes),
                     "region_counts": {k: int(v.sum()) for k, v in regions.items()},
                     "methods": {}}

    # ---------------- moment residuals (global frame for x/y/lap) -------------------
    logger.info("")
    logger.info("Moment residual RMS, Taylor consistency to order %d (0 = exact)", order)
    hdr = "  %-22s %-8s" + " %14s" * 3
    logger.info(hdr, "method", "op", *regions)
    for name, w in methods.items():
        W = np.stack([w["x"], w["y"], w["laplace"]], axis=-1)
        res = moment_residual_batch(W, off_glob, ops, order)
        # scale to normalised units so x/y and laplace are comparable across h
        scale = np.stack([h, h, h ** 2], axis=-1)[:, None, :]
        rms = np.sqrt(((res * scale) ** 2).mean(axis=1))          # (S, 3)
        rec = results["methods"].setdefault(name, {})
        for k, op in enumerate(ops):
            vals = [float(rms[m, k].mean()) for m in regions.values()]
            rec.setdefault("moment_rms", {})[op] = dict(zip(regions, vals))
            logger.info("  %-22s %-8s" + " %14.3e" * 3, name, op, *vals)

    # ---------------- derivative errors on analytic fields --------------------------
    xs, ys = cloud.coords[:, 0], cloud.coords[:, 1]
    for fname, u, ux, uy, lap in FIELDS:
        u_st = u(cloud.coords[nb][..., 0], cloud.coords[nb][..., 1])
        exact = {"x": ux(xs, ys), "y": uy(xs, ys), "laplace": lap(xs, ys)}
        nrm = cloud.normals
        dudn = nrm[:, 0] * exact["x"] + nrm[:, 1] * exact["y"]
        logger.info("")
        logger.info("Relative L2 error, u = %s", fname)
        logger.info(hdr, "method", "op", *regions)
        for name, w in methods.items():
            rec = results["methods"][name].setdefault("rel_l2", {}).setdefault(fname, {})
            for op in ops:
                pred = apply(w[op], u_st)
                vals = [rel_l2(pred[m], exact[op][m]) for m in regions.values()]
                rec[op] = dict(zip(regions, vals))
                logger.info("  %-22s %-8s" + " %14.3e" * 3, name, op, *vals)
            pn = apply(w["normal"][wall], u_st[wall])
            rec["normal_wall"] = rel_l2(pn, dudn[wall])
        logger.info("  Normal derivative at the %d wall nodes (Neumann/Robin row):", wall.sum())
        for name in methods:
            logger.info("    %-22s rel L2 %.3e", name,
                        results["methods"][name]["rel_l2"][fname]["normal_wall"])

    # ---------------- weight magnitudes (Laplacian) -------------------------------
    logger.info("")
    logger.info("Largest |w| x h^2, Laplacian (normalised; large = oscillatory, noise-amplifying)")
    logger.info("  %-22s" + " %14s" * 3, "method", *regions)
    for name, w in methods.items():
        wl = np.abs(w["laplace"]) * h[:, None] ** 2
        vals = [float(wl[m].max()) for m in regions.values()]
        results["methods"][name]["max_abs_w_lap_norm"] = dict(zip(regions, vals))
        logger.info("  %-22s" + " %14.3g" * 3, name, *vals)

    # ---------------- diagonal sign of the Laplacian (centre weight) ---------------
    logger.info("")
    logger.info("Laplacian rows with a non-negative centre weight w_ii (expected on one-sided wall"
                " stencils, a warning sign in the interior)")
    for name, w in methods.items():
        wii = -w["laplace"][:, 1:].sum(axis=1)
        frac = [float((wii[m] >= 0).mean()) for m in regions.values()]
        results["methods"][name]["frac_bad_centre"] = dict(zip(regions, frac))
        logger.info("  %-22s" + " %13.1f%%" * 3, name, *[100 * f for f in frac])

    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("")
        logger.info("Wrote %s", out_json)
        # per-node Laplacian error on the first field, for plotting
        _, u, _, _, lap = FIELDS[0]
        u_st = u(cloud.coords[nb][..., 0], cloud.coords[nb][..., 1])
        errs = {name.replace(" ", "_").replace("/", "_"): apply(w["laplace"], u_st) - lap(xs, ys)
                for name, w in methods.items()}
        np.savez(jn(os.path.dirname(out_json), "laplace_error_fields.npz"),
                 coords=cloud.coords, wall=wall, near=near, **errs)
    return results


if __name__ == "__main__":
    evaluate()
