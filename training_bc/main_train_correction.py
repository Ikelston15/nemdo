"""Stage 4 -- train a data-driven correction on solution fields.

Stages 1-3 produce an operator that is *consistent*: it differentiates every
polynomial of degree <= p correctly. That is also the ceiling of what it can be,
because the classical local solve satisfies the same conditions exactly. To do better
the operator has to know something a polynomial constraint does not -- namely which
fields the PDE actually produces.

This stage learns that, without giving up the first part. The correction is projected
onto ``null(A)``, so the corrected operator reproduces polynomials exactly as well as
the base does, and the convergence order is untouched. What the correction changes is
the choice *within* the set of consistent operators: the classical weighted solve
picks the minimum-norm point of that set, and the network picks the point that works
best on your solutions.

Default base is the classical solve, not the learned operator (see
``models/nemdo_correction.py`` for why).

Run:
    python main_build_dataset.py && python main_train_correction.py
"""

from __future__ import annotations

import logging
import os
import time
from os.path import join as jn

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from functions.correction import (apply_correction, base_weights_normalised,
                                  correction_features, operator_values)
from functions.domains import default_domain_mix
from functions.fields import gather_on_stencils, sample_fields
from functions.moments import monomial_basis
from functions.stencils import StencilBatcher, StencilSet, concat, extract_stencils
from models.nemdo_bc import load_bc_model
from models.nemdo_correction import NEMDO_Correction, save_correction_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_field_data(stencils: StencilSet, clouds, n_fields: int, families, seed: int,
                     ppw_range: tuple[float, float]):
    """Sample fields per cloud and expand them onto the stencils.

    Stencils from different clouds index different node sets, so fields are drawn per
    cloud and stitched back together in the original stencil order.
    """
    u_all = np.zeros((len(stencils), n_fields, stencils.stencil_size), dtype=np.float32)
    lap_all = np.zeros((len(stencils), n_fields), dtype=np.float32)

    for ci, cloud in enumerate(clouds):
        mask = stencils.cloud_index == ci
        if not mask.any():
            continue
        sample = sample_fields(cloud.coords, cloud.spacing, n_fields,
                               families=families, seed=seed + 1000 * ci,
                               fourier={"ppw_range": ppw_range})
        u_st, lap = gather_on_stencils(sample, stencils.neighbour_index[mask])
        u_all[mask] = u_st
        lap_all[mask] = lap
    return u_all, lap_all


def run_epoch(model, loader, stencils, base_norm, u_data, lap_data, basis, device,
              optimizer=None, delta_penalty: float = 1e-3, power: int = 2):
    """One pass over the stencils; every stencil is scored against all fields."""
    mon_power, inv_factorial = basis
    training = optimizer is not None
    model.train(training)

    k = stencils.stencil_size
    tot_mse = 0.0
    tot_n = 0
    tot_wall = 0.0
    n_wall = 0

    for x, edge_index, batch, sel in loader:
        b = len(sel)
        bw = torch.from_numpy(base_norm[sel]).to(device)
        h = torch.from_numpy(stencils.h[sel]).to(device)
        u = torch.from_numpy(u_data[sel]).to(device)
        target = torch.from_numpy(lap_data[sel]).to(device)

        with torch.set_grad_enabled(training):
            raw = model(x, edge_index, batch)
            w, delta = apply_correction(raw, x, bw, mon_power, inv_factorial, b, k)
            pred = operator_values(w, u, h, power=power)
            err = (pred - target) ** 2
            loss = err.mean()
            if delta_penalty > 0:
                loss = loss + delta_penalty * (delta ** 2).mean()

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        e = err.detach().mean(dim=1).cpu().numpy()
        tot_mse += float(e.sum())
        tot_n += b
        is_wall = stencils.is_boundary[sel]
        tot_wall += float(e[is_wall].sum())
        n_wall += int(is_wall.sum())

    return tot_mse / max(tot_n, 1), tot_wall / max(n_wall, 1)


def train(data_dir: str,
          base_model_path: str | None,
          base_kind: str,
          out_path: str,
          model_id: int,
          spacing: float,
          stencil_size: int,
          n_fields: int,
          families: tuple[str, ...],
          ppw_range: tuple[float, float],
          epochs: int,
          batch_size: int,
          lr: float,
          layers: int,
          embedding_size: int,
          approximation_order: int,
          projection_order: int,
          delta_penalty: float,
          seed: int) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    os.makedirs(out_path, exist_ok=True)

    # Rebuild the clouds so field sampling has the coordinates, then re-extract the
    # stencils from them. Cheap, and it keeps cloud and stencil exactly in step.
    clouds = default_domain_mix(spacing=spacing, seed=1)
    stencils = concat([extract_stencils(c, stencil_size=stencil_size, cloud_index=i)
                       for i, c in enumerate(clouds)])
    logger.info("%d stencils from %d clouds (%.1f%% on a wall)",
                len(stencils), len(clouds), 100 * stencils.is_boundary.mean())

    base_model = None
    if base_kind == "nemdo":
        base_model, _ = load_bc_model(base_model_path, device=device)
    base_norm = base_weights_normalised(stencils, kind=base_kind, base_model=base_model,
                                        operator="laplace", order=approximation_order,
                                        device=device)
    logger.info("Base operator: %s (max |w_norm| = %.3g)", base_kind, np.abs(base_norm).max())

    feats = correction_features(stencils, base_norm)
    stencils.features = feats                      # the model sees the base weight too

    logger.info("Sampling %d fields per cloud from %s, PPW in [%.0f, %.0f]",
                n_fields, list(families), *ppw_range)
    u_tr, lap_tr = build_field_data(stencils, clouds, n_fields, families, seed, ppw_range)
    u_va, lap_va = build_field_data(stencils, clouds, max(n_fields // 4, 4), families,
                                    seed + 77_000, ppw_range)

    model = NEMDO_Correction(embedding_size=embedding_size, layers=layers,
                             input_size=feats.shape[-1]).to(device)
    logger.info("Correction model: %d parameters",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    plateau = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, cooldown=5, eps=1e-12)

    # Project against HIGHER-order moments than the base is consistent to. Constraining
    # only order 2 leaves the third-order moments free, and those carry the O(h)
    # truncation term -- the correction then buys accuracy at the training resolution
    # by spending convergence. Measured: see the module docstring in
    # functions/projection.py.
    basis = monomial_basis(projection_order, device=device)
    from functions.projection import null_space_dimension
    logger.info("Projection: order %d (base consistency %d) -- correction has %d of %d "
                "dimensions to use", projection_order, approximation_order,
                null_space_dimension(stencil_size, projection_order), stencil_size)

    train_loader = StencilBatcher(stencils, batch_size, shuffle=True, drop_last=True,
                                  device=device, seed=seed)
    val_loader = StencilBatcher(stencils, batch_size, device=device)

    # Where we start: the untrained correction is exactly zero, so this is the base.
    base_mse, base_wall = run_epoch(model, val_loader, stencils, base_norm, u_va, lap_va,
                                    basis, device, delta_penalty=0.0)
    logger.info("Base operator alone: val MSE %.4e (wall %.4e)", base_mse, base_wall)

    history = {"train": [], "val": [], "val_wall": []}
    best = np.inf
    best_state = None
    save_path = jn(out_path, f"nemdo_correction_{model_id}.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr, _ = run_epoch(model, train_loader, stencils, base_norm, u_tr, lap_tr, basis,
                          device, optimizer=optimizer, delta_penalty=delta_penalty)
        va, va_wall = run_epoch(model, val_loader, stencils, base_norm, u_va, lap_va,
                                basis, device, delta_penalty=0.0)

        history["train"].append(float(tr))
        history["val"].append(float(va))
        history["val_wall"].append(float(va_wall))

        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        plateau.step(tr)
        logger.info("Epoch %3d | train %.4e | val %.4e (wall %.4e) | vs base x%.3f | %.1fs",
                    epoch, tr, va, va_wall, va / max(base_mse, 1e-30),
                    time.perf_counter() - t0)

    if best_state is not None:
        model.load_state_dict(best_state)
    save_correction_model(save_path, model, optimizer, history=history, epochs=epochs,
                          base=base_kind, base_model_path=base_model_path,
                          approximation_order=approximation_order,
                          projection_order=projection_order,
                          stencil_size=stencil_size, spacing=spacing,
                          families=list(families), ppw_range=list(ppw_range),
                          best_val_mse=float(best), base_val_mse=float(base_mse),
                          model_id=model_id)
    logger.info("Best val MSE %.4e vs base %.4e -- improvement x%.2f -- saved %s",
                best, base_mse, base_mse / max(best, 1e-30), save_path)
    return save_path


if __name__ == "__main__":
    data_dir            = "preproc_data_bc"
    base_model_path     = jn("saved_models_bc", "nemdo_bc_1.pth")
    base_kind           = "classical"   # 'classical' or 'nemdo'
    out_path            = "saved_models_bc"
    model_id            = 1

    spacing             = 0.022         # must match main_build_dataset.py
    stencil_size        = 35
    approximation_order = 2         # consistency order of the BASE operator
    projection_order    = 3         # moments the correction may not touch (see README:
                                    #   2 = big gain, convergence destroyed
                                    #   3 = real gain, convergence intact  <- default
                                    #   4 = convergence intact, gain gone

    n_fields            = 24            # solution fields sampled per cloud
    families            = ("fourier", "bumps", "layer")
    ppw_range           = (8.0, 24.0)   # points per wavelength: the resolution knob

    epochs              = 60
    batch_size          = 256
    lr                  = 1e-3
    layers              = 2
    embedding_size      = 128
    delta_penalty       = 1e-3          # keeps the correction small
    seed                = 1

    train(data_dir=data_dir, base_model_path=base_model_path, base_kind=base_kind,
          out_path=out_path, model_id=model_id, spacing=spacing,
          stencil_size=stencil_size, n_fields=n_fields, families=families,
          ppw_range=ppw_range, epochs=epochs, batch_size=batch_size, lr=lr,
          layers=layers, embedding_size=embedding_size,
          approximation_order=approximation_order,
          projection_order=projection_order, delta_penalty=delta_penalty,
          seed=seed)
