"""Boundary-aware NeMDO model.

The trunk is the message-passing architecture from ``training/models/NEMDO_mod.py``
(attention aggregation over a star graph, tanh MLPs) -- the geometry encoder is not
what needed changing.  Two things are different:

* ``input_size`` is 4 rather than 2.  Alongside ``dx/h, dy/h`` each node carries an
  ``is_centre`` flag and the stencil's boundary proximity, so the network can tell a
  one-sided wall stencil from an interior one that merely looks lopsided after
  jitter.

* There is one decoder head per operator on a shared trunk.  Training ``d/dx``,
  ``d/dy`` and the Laplacian together means a single run produces everything needed
  to assemble interior rows *and* Neumann/Robin rows, and the heads share whatever
  the trunk learns about one-sided geometry -- which is the part that is scarce in
  the data.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import AttentionalAggregation

logger = logging.getLogger(__name__)


def reset_params(layers, activation: str) -> None:
    for layer in layers:
        if isinstance(layer, nn.Linear):
            if activation == "selu":
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(activation))
                nn.init.zeros_(layer.bias)
            elif activation == "tanh":
                nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
                nn.init.zeros_(layer.bias)


class GraphLayer(MessagePassing):
    """Message passing with attentional aggregation (as in ``NEMDO_mod.GraphLayer``)."""

    def __init__(self, embedding_size: int):
        super().__init__(aggr=None)

        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),
        )

        self.attention_aggr = AttentionalAggregation(
            nn.Sequential(nn.Linear(embedding_size, embedding_size)),
            self.attention_mlp,
        )

        self.mlp_msg = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),
        )
        reset_params(self.mlp_msg, "tanh")

        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * embedding_size, embedding_size),
            nn.Tanh(),
        )
        reset_params(self.mlp_upd, "tanh")

    def forward(self, node_feature: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        return self.propagate(edge_index, node_feature=node_feature, batch=batch)

    def aggregate(self, mes: Tensor, index: Tensor) -> Tensor:
        return self.attention_aggr(x=mes, index=index)

    def message(self, node_feature_i: Tensor, node_feature_j: Tensor, batch: Tensor) -> Tensor:
        return self.mlp_msg(node_feature_j)

    def update(self, aggr: Tensor, node_feature: Tensor, batch: Tensor) -> Tensor:
        return self.mlp_upd(torch.cat((node_feature, aggr), dim=1))


class NEMDO_BC(nn.Module):
    """Shared geometry trunk, one weight head per differential operator."""

    def __init__(self,
                 embedding_size: int,
                 layers: int,
                 operators: Sequence[str] = ("x", "y", "laplace"),
                 input_size: int = 4):
        super().__init__()

        self.embedding_size = embedding_size
        self.operators = tuple(operators)
        self.input_size = input_size
        hidden = embedding_size // 2

        self.node_encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Tanh(),
        )
        reset_params(self.node_encoder, "tanh")

        self.graph_layers = nn.ModuleList(GraphLayer(hidden) for _ in range(layers))

        self.heads = nn.ModuleList()
        for _ in self.operators:
            head = nn.Sequential(
                nn.Linear(hidden, embedding_size),
                nn.Tanh(),
                nn.Linear(embedding_size, 1),
            )
            reset_params(head, "tanh")
            self.heads.append(head)

    def forward(self, node_feature: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Returns (N, n_operators) normalised weights, one column per operator."""
        emb = self.node_encoder(node_feature)
        for layer in self.graph_layers:
            emb = layer(emb, edge_index, batch)
        return torch.cat([head(emb) for head in self.heads], dim=1)


def save_bc_model(path: str, model: NEMDO_BC, optimizer, **extra) -> None:
    """Checkpoint in the same shape ``training/`` uses, plus the operator list."""
    payload = {
        "weights": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "embedding_size": model.embedding_size,
        "layers": len(model.graph_layers),
        "input_size": model.input_size,
        "operators": list(model.operators),
    }
    payload.update(extra)
    torch.save(payload, path)


def load_bc_model(path: str, device: str = "cpu") -> tuple[NEMDO_BC, dict]:
    """Rebuild a model from a checkpoint written by ``save_bc_model``."""
    attrs = torch.load(path, map_location=device, weights_only=False)
    model = NEMDO_BC(embedding_size=attrs["embedding_size"],
                     layers=attrs["layers"],
                     operators=attrs["operators"],
                     input_size=attrs.get("input_size", 4))

    state = OrderedDict(
        (k[len("module."):], v) if k.startswith("module.") else (k, v)
        for k, v in attrs["weights"].items())
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info("Loaded boundary-aware model from %s (operators=%s)", path, attrs["operators"])
    return model, attrs
