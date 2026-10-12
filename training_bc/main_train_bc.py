"""Stage 2 -- train the boundary-aware operator.

Equivalent to ``training/main_train_gpu.py``, with four changes:

* the dataset is the boundary-containing one from ``main_build_dataset.py``;
* one model learns several operators at once (``d/dx``, ``d/dy``, Laplacian), so a
  single run yields everything needed for interior rows *and* Neumann/Robin rows;
* the loss reweights by boundary proximity, so wall stencils are not drowned out by
  the far more numerous interior ones;
* a weight-norm penalty keeps one-sided stencils from drifting to large oscillatory
  weights, which is what the classical minimum-norm solve avoids by construction.

Run:
    python main_build_dataset.py && python main_train_bc.py
"""

from __future__ import annotations

import logging
import os
import time
from os.path import join as jn

import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from functions.moments import (DEFAULT_OPERATORS, calc_moments_multihead,
                               consistency_loss, monomial_basis, target_moments)
from functions.stencils import StencilBatcher, StencilSet
from models.nemdo_bc import NEMDO_BC, save_bc_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sample_weights(proximity: np.ndarray, boundary_weight: float) -> torch.Tensor:
    """1 deep inside, ``boundary_weight`` on the wall, linear in between."""
    return torch.from_numpy(
        (1.0 + (boundary_weight - 1.0) * proximity).astype(np.float32))


def run_epoch(model, loader, basis, target, device, optimizer=None,
              weight_penalty: float = 1e-4, boundary_weight: float = 3.0,
              proximity: np.ndarray | None = None):
    """One pass. Returns (weighted loss, unweighted interior MSE, unweighted wall MSE)."""
    mon_power, inv_factorial = basis
    training = optimizer is not None
    model.train(training)

    totals = np.zeros(3)      # sum(loss), sum(mse * n) interior, sum(mse * n) wall
    counts = np.zeros(3)

    for x, edge_index, batch, sel in loader:
        n_graphs = len(sel)
        prox = proximity[sel]
        sw = sample_weights(prox, boundary_weight).to(device)

        with torch.set_grad_enabled(training):
            out = model(x, edge_index, batch)
            moments = calc_moments_multihead(x, out, batch, mon_power, inv_factorial, n_graphs)
            loss, per_graph = consistency_loss(moments, target, out, batch, n_graphs,
                                               sample_weight=sw,
                                               weight_penalty=weight_penalty)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pg = per_graph.cpu().numpy()
        totals[0] += float(loss.detach()) * n_graphs
        counts[0] += n_graphs
        is_wall = prox > 0.0
        totals[1] += pg[~is_wall].sum()
        counts[1] += int((~is_wall).sum())
        totals[2] += pg[is_wall].sum()
        counts[2] += int(is_wall.sum())

    safe = np.where(counts > 0, counts, 1)
    out = totals / safe
    return out[0], out[1], out[2]


def train(data_dir: str,
          out_path: str,
          checkpoint_path: str,
          model_id: int,
          operators: tuple[str, ...],
          epochs: int,
          batch_size: int,
          lr: float,
          layers: int,
          embedding_size: int,
          approximation_order: int,
          weight_penalty: float,
          boundary_weight: float,
          checkpoint_every: int,
          seed: int) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    train_set = StencilSet.load(jn(data_dir, "train.npz"))
    val_set = StencilSet.load(jn(data_dir, "val.npz"))
    logger.info("Train %d stencils (%.1f%% near a boundary), val %d, device %s",
                len(train_set), 100 * (train_set.proximity > 0).mean(), len(val_set), device)

    model = NEMDO_BC(embedding_size=embedding_size, layers=layers,
                     operators=operators,
                     input_size=train_set.features.shape[-1]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d parameters, heads %s", n_params, operators)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=min(10, epochs))
    plateau = ReduceLROnPlateau(optimizer, patience=12, factor=0.4, cooldown=8, eps=1e-12)

    basis = monomial_basis(approximation_order, device=device)
    target = torch.from_numpy(target_moments(operators, approximation_order)).to(device)

    train_loader = StencilBatcher(train_set, batch_size, shuffle=True, drop_last=True,
                                  device=device, seed=seed)
    val_loader = StencilBatcher(val_set, batch_size, device=device)

    history = {"train": [], "val": [], "val_interior": [], "val_wall": []}
    best = np.inf
    best_state = None
    save_path = jn(out_path, f"nemdo_bc_{model_id}.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_in, tr_wall = run_epoch(
            model, train_loader, basis, target, device, optimizer=optimizer,
            weight_penalty=weight_penalty, boundary_weight=boundary_weight,
            proximity=train_set.proximity)
        va_loss, va_in, va_wall = run_epoch(
            model, val_loader, basis, target, device,
            weight_penalty=weight_penalty, boundary_weight=boundary_weight,
            proximity=val_set.proximity)

        history["train"].append(float(tr_loss))
        history["val"].append(float(va_loss))
        history["val_interior"].append(float(va_in))
        history["val_wall"].append(float(va_wall))

        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        plateau.step(tr_loss)
        if epoch <= warmup.total_iters:
            warmup.step()

        logger.info("Epoch %3d | train %.4e | val %.4e | val moment MSE: interior %.3e "
                    "wall %.3e | %.1fs", epoch, tr_loss, va_loss, va_in, va_wall,
                    time.perf_counter() - t0)

        if checkpoint_every and epoch % checkpoint_every == 0:
            save_bc_model(jn(checkpoint_path, f"nemdo_bc_{model_id}_epoch{epoch}.pth"),
                          model, optimizer, history=history, epochs=epoch,
                          approximation_order=approximation_order,
                          stencil_size=train_set.stencil_size, best_val_loss=float(best))

    if best_state is not None:
        model.load_state_dict(best_state)
    save_bc_model(save_path, model, optimizer, history=history, epochs=epochs,
                  approximation_order=approximation_order,
                  stencil_size=train_set.stencil_size, best_val_loss=float(best),
                  batch_size=batch_size, lr=lr, model_id=model_id)
    logger.info("Best val loss %.4e -- saved %s", best, save_path)
    return save_path


if __name__ == "__main__":
    data_dir            = "preproc_data_bc"
    base_model_path     = "saved_models_bc"
    model_id            = 1
    operators           = DEFAULT_OPERATORS   # ('x', 'y', 'laplace')
    epochs              = 300
    batch_size          = 256
    lr                  = 1e-3
    layers              = 3
    embedding_size      = 192      # ~196k parameters; ~8.5 s/epoch on CPU, far less on GPU
    approximation_order = 2        # weights exact for polynomials up to this degree
    weight_penalty      = 1e-4     # pulls toward the minimum-norm solution
    boundary_weight     = 3.0      # loss multiplier at the wall relative to deep interior
    checkpoint_every    = 0        # 0 disables intermediate checkpoints
    seed                = 1

    train(data_dir=data_dir,
          out_path=base_model_path,
          checkpoint_path=jn(base_model_path, "checkpoint"),
          model_id=model_id,
          operators=operators,
          epochs=epochs,
          batch_size=batch_size,
          lr=lr,
          layers=layers,
          embedding_size=embedding_size,
          approximation_order=approximation_order,
          weight_penalty=weight_penalty,
          boundary_weight=boundary_weight,
          checkpoint_every=checkpoint_every,
          seed=seed)
