"""Stage 5 -- does the correction actually help, and what does it cost?

Four questions, in order of how easy they are to fool yourself about.

1. **In distribution.** On held-out fields of the kind it was trained on, is the
   corrected operator better than the classical solve? This is the claim.

2. **Out of distribution.** On fields it was *not* trained for -- much coarser or much
   smoother than the training band -- does it still help, or does it hurt? This is the
   price of specialising, and reporting it is the difference between a result and a
   sales pitch.

3. **Polynomial exactness.** The projection should make this a non-event: the
   corrected operator must differentiate polynomials exactly as well as the base. If
   this moves, the projection is broken.

4. **Convergence under refinement.** The real risk with any data-driven
   discretisation is an operator that looks good on the mesh it was tuned for and then
   stops improving. Taking a *fixed* physical field and refining the node spacing,
   the corrected operator must converge at the same order as the base.

Run:
    python main_train_correction.py && python main_eval_correction.py
"""

from __future__ import annotations

import logging
from os.path import join as jn

import numpy as np
import torch

from functions.correction import (apply_correction, apply_numpy,
                                  base_weights_normalised, correction_features)
from functions.domains import holdout_domain
from functions.fields import gather_on_stencils, sample_fields
from functions.moments import monomial_basis
from functions.stencils import StencilBatcher, extract_stencils
from models.nemdo_bc import load_bc_model
from models.nemdo_correction import load_correction_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def corrected_weights(model, stencils, base_norm, order: int, device: str = "cpu",
                      batch_size: int = 1024) -> np.ndarray:
    """Run the correction over a stencil set.  Returns normalised weights (S, K).

    ``order`` here is the **projection** order the model was trained with, not the
    base's consistency order; using the wrong one silently changes which moments the
    correction is allowed to move.
    """
    feats = correction_features(stencils, base_norm)
    original = stencils.features
    stencils.features = feats
    try:
        basis = monomial_basis(order, device=device)
        out = np.empty_like(base_norm)
        loader = StencilBatcher(stencils, batch_size=batch_size, device=device)
        with torch.no_grad():
            for x, edge_index, batch, sel in loader:
                b = len(sel)
                bw = torch.from_numpy(base_norm[sel]).to(device)
                raw = model(x, edge_index, batch)
                w, _ = apply_correction(raw, x, bw, basis[0], basis[1], b,
                                        stencils.stencil_size)
                out[sel] = w.cpu().numpy()
        return out
    finally:
        stencils.features = original


def _rel_l2(pred: np.ndarray, exact: np.ndarray) -> float:
    d = np.linalg.norm(exact)
    return float(np.linalg.norm(pred - exact) / d) if d > 0 else float("nan")


def _score(weights: np.ndarray, u_st: np.ndarray, h: np.ndarray,
           target: np.ndarray) -> tuple[float, float]:
    """Relative L2 error over all (stencil, field) pairs, and the same at wall nodes."""
    pred = apply_numpy(weights.astype(np.float64), u_st.astype(np.float64),
                       h.astype(np.float64))
    return pred, _rel_l2(pred, target)


def evaluate(correction_path: str, base_bc_path: str | None, spacing: float,
             stencil_size: int, order: int = 2, n_fields: int = 24,
             device: str = "cpu") -> None:
    model, attrs = load_correction_model(correction_path, device=device)
    base_kind = attrs.get("base", "classical")
    train_ppw = tuple(attrs.get("ppw_range", (8.0, 24.0)))
    families = tuple(attrs.get("families", ("fourier", "bumps", "layer")))
    proj_order = int(attrs.get("projection_order", order))
    logger.info("Correction: base=%s, trained on PPW %s, families %s, %d epochs",
                base_kind, train_ppw, list(families), attrs.get("epochs", -1))
    logger.info("Base consistency order %d; correction projected against order %d",
                order, proj_order)
    logger.info("Training-time val MSE %.4e vs base %.4e (x%.2f)",
                attrs.get("best_val_mse", np.nan), attrs.get("base_val_mse", np.nan),
                attrs.get("base_val_mse", np.nan) / max(attrs.get("best_val_mse", 1), 1e-30))

    base_model = None
    if base_kind == "nemdo" or base_bc_path:
        try:
            base_model, _ = load_bc_model(base_bc_path, device=device)
        except Exception as exc:                      # evaluation should still run
            logger.warning("Could not load the NeMDO base model (%s); skipping it", exc)

    cloud = holdout_domain(spacing=spacing)
    st = extract_stencils(cloud, stencil_size=stencil_size, cloud_index=-1)
    logger.info("Holdout '%s': %d nodes, %d on the boundary (shape not in training)",
                cloud.name, cloud.n_nodes, int(cloud.is_boundary.sum()))

    base_norm = base_weights_normalised(st, kind=base_kind, base_model=base_model,
                                        operator="laplace", order=order, device=device)
    corr_norm = corrected_weights(model, st, base_norm, proj_order, device=device)

    nemdo_norm = None
    if base_model is not None and base_kind != "nemdo":
        nemdo_norm = base_weights_normalised(st, kind="nemdo", base_model=base_model,
                                             operator="laplace", order=order, device=device)

    wall = st.is_boundary

    def report(title: str, seed: int, ppw: tuple[float, float],
               fams: tuple[str, ...] = families) -> None:
        sample = sample_fields(cloud.coords, cloud.spacing, n_fields, families=fams,
                               seed=seed, fourier={"ppw_range": ppw})
        u_st, target = gather_on_stencils(sample, st.neighbour_index)
        rows = [(f"base ({base_kind})", base_norm), ("corrected", corr_norm)]
        if nemdo_norm is not None:
            rows.append(("nemdo stage-2", nemdo_norm))
        logger.info("")
        logger.info("%s  [PPW %.0f-%.0f]", title, *ppw)
        logger.info("  %-18s %12s %12s", "operator", "rel L2", "wall only")
        ref = None
        for name, w in rows:
            pred, err = _score(w, u_st, st.h, target)
            _, werr = _score(w[wall], u_st[wall], st.h[wall], target[wall])
            if name.startswith("base"):
                ref = err
            tag = "" if ref is None or name.startswith("base") else f"   (x{ref / max(err, 1e-30):.2f} vs base)"
            logger.info("  %-18s %12.3e %12.3e%s", name, err, werr, tag)

    report("1. IN DISTRIBUTION -- held-out fields of the trained kind", 4242, train_ppw)

    coarse = (max(train_ppw[0] * 0.4, 3.0), train_ppw[0])
    smooth = (train_ppw[1] * 2.0, train_ppw[1] * 5.0)
    report("2a. OUT OF DISTRIBUTION -- coarser than trained", 5151, coarse)
    report("2b. OUT OF DISTRIBUTION -- much smoother than trained", 6161, smooth)

    # ---- 3. polynomial exactness -------------------------------------------------
    sample = sample_fields(cloud.coords, cloud.spacing, 8, families=("polynomial",),
                           seed=909, normalise=False)
    u_st, target = gather_on_stencils(sample, st.neighbour_index)
    logger.info("")
    logger.info("3. POLYNOMIAL EXACTNESS (degree %d; the projection should keep this "
                "identical to the base)", order)
    _, e_base = _score(base_norm, u_st, st.h, target)
    _, e_corr = _score(corr_norm, u_st, st.h, target)
    logger.info("  base      rel L2 %.3e", e_base)
    logger.info("  corrected rel L2 %.3e   (shift %.1e)", e_corr, abs(e_corr - e_base))

    # ---- 4. refinement -------------------------------------------------------------
    logger.info("")
    logger.info("4. CONVERGENCE on a FIXED physical field under refinement")
    logger.info("  %-10s %8s %14s %8s %14s %8s", "spacing", "nodes",
                "base rel L2", "order", "corr rel L2", "order")
    u_fn = lambda p: np.sin(3 * p[:, 0]) * np.cos(2 * p[:, 1]) + p[:, 0] ** 3
    lap_fn = lambda p: -13 * np.sin(3 * p[:, 0]) * np.cos(2 * p[:, 1]) + 6 * p[:, 0]

    spacings = [spacing * f for f in (2.0, 1.5, 1.0, 0.72)]
    prev = None
    for s in spacings:
        c = holdout_domain(spacing=s)
        sc = extract_stencils(c, stencil_size=stencil_size, cloud_index=-1)
        bn = base_weights_normalised(sc, kind=base_kind, base_model=base_model,
                                     operator="laplace", order=order, device=device)
        cn = corrected_weights(model, sc, bn, proj_order, device=device)
        u = u_fn(c.coords)[None, :]
        u_s = np.ascontiguousarray(u[:, sc.neighbour_index].transpose(1, 0, 2))
        tgt = lap_fn(c.coords)[sc.neighbour_index[:, 0]][:, None]
        _, eb = _score(bn, u_s, sc.h, tgt)
        _, ec = _score(cn, u_s, sc.h, tgt)
        if prev is None:
            logger.info("  %-10.4f %8d %14.3e %8s %14.3e %8s", s, c.n_nodes, eb, "-", ec, "-")
        else:
            r = np.log(prev[0] / s)
            logger.info("  %-10.4f %8d %14.3e %8.2f %14.3e %8.2f", s, c.n_nodes,
                        eb, np.log(prev[1] / eb) / r, ec, np.log(prev[2] / ec) / r)
        prev = (s, eb, ec)
    logger.info("  (a corrected order materially below the base order would mean the "
                "correction is trading convergence for coarse-mesh accuracy)")


if __name__ == "__main__":
    correction_path = jn("saved_models_bc", "nemdo_correction_1.pth")
    base_bc_path    = jn("saved_models_bc", "nemdo_bc_1.pth")
    spacing         = 0.022 * 0.9
    stencil_size    = 35
    order           = 2
    n_fields        = 24

    evaluate(correction_path=correction_path, base_bc_path=base_bc_path,
             spacing=spacing, stencil_size=stencil_size, order=order, n_fields=n_fields)
