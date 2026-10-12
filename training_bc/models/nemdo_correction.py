"""Data-driven correction on top of a consistency-satisfying base operator.

The division of labour:

* the **base operator** supplies consistency -- it differentiates polynomials of
  degree <= p correctly, which is what makes the scheme converge under refinement;
* the **correction** supplies everything consistency does not determine, fitted to the
  solutions the PDE actually produces.

The correction is projected onto ``null(A)`` (see ``functions/projection.py``), so it
*cannot* damage the first part no matter what it learns. Consistency is structural
here, not something the loss has to be trusted to preserve.

The base can be either the classical weighted least-squares solve or a trained NeMDO
operator. Classical is the default, and that is a deliberate choice: producing
consistent weights is a solved problem that a local solve does exactly and for free,
so there is little point spending network capacity on reproducing it. What a classical
method has no way to express is which point of the consistent set suits *these*
solutions -- and that is exactly what the correction learns. Use a NeMDO base instead
when the local solve is the thing you are trying to avoid, e.g. nodes that move every
time step.

The final layer is zero-initialised, so training starts from exactly the base
operator and can only move away from it deliberately.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from torch import Tensor, nn

from models.nemdo_bc import GraphLayer, reset_params

logger = logging.getLogger(__name__)

# [dx/h, dy/h, is_centre, boundary_proximity, base_weight]
N_CORRECTION_FEATURES = 5


class NEMDO_Correction(nn.Module):
    """Per-node weight correction, conditioned on the stencil and its base weights."""

    def __init__(self,
                 embedding_size: int,
                 layers: int,
                 input_size: int = N_CORRECTION_FEATURES):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        hidden = embedding_size // 2

        self.node_encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Tanh(),
        )
        reset_params(self.node_encoder, "tanh")

        self.graph_layers = nn.ModuleList(GraphLayer(hidden) for _ in range(layers))

        self.decoder = nn.Sequential(
            nn.Linear(hidden, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, 1),
        )
        reset_params(self.decoder, "tanh")

        # Start as an exact no-op: at initialisation the corrected operator *is* the
        # base operator, so any later difference is something training chose.
        final = self.decoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, node_feature: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Returns (N, 1) raw per-node corrections, before null-space projection."""
        emb = self.node_encoder(node_feature)
        for layer in self.graph_layers:
            emb = layer(emb, edge_index, batch)
        return self.decoder(emb)


def save_correction_model(path: str, model: NEMDO_Correction, optimizer, **extra) -> None:
    payload = {
        "weights": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "embedding_size": model.embedding_size,
        "layers": len(model.graph_layers),
        "input_size": model.input_size,
    }
    payload.update(extra)
    torch.save(payload, path)


def load_correction_model(path: str, device: str = "cpu") -> tuple[NEMDO_Correction, dict]:
    attrs = torch.load(path, map_location=device, weights_only=False)
    model = NEMDO_Correction(embedding_size=attrs["embedding_size"],
                             layers=attrs["layers"],
                             input_size=attrs.get("input_size", N_CORRECTION_FEATURES))
    state = OrderedDict(
        (k[len("module."):], v) if k.startswith("module.") else (k, v)
        for k, v in attrs["weights"].items())
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info("Loaded correction model from %s (base=%s)", path, attrs.get("base", "?"))
    return model, attrs
