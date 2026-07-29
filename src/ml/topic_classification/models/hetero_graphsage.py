"""PyTorch Geometric heterogeneous GraphSAGE topic-link predictor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F  # noqa: N812 - PyTorch convention
from torch import Tensor, nn
from torch_geometric.nn import HeteroConv, SAGEConv

from ml.topic_classification.graph import EdgeType


class HeterogeneousGraphSAGE(nn.Module):
    """Contextualize typed KG nodes and score every article-topic pair."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        edge_types: Sequence[EdgeType],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        required_types = {"article", "entity", "topic", "domain"}
        missing = required_types - set(input_dims)
        if missing:
            values = ", ".join(sorted(missing))
            raise ValueError(f"Missing input dimensions for node types: {values}")

        self.input_dims = dict(input_dims)
        self.edge_types = tuple(edge_types)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.projections = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dim, hidden_dim)
                for node_type, input_dim in self.input_dims.items()
            }
        )
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (hidden_dim, hidden_dim),
                            hidden_dim,
                            aggr="mean",
                        )
                        for edge_type in self.edge_types
                    },
                    aggr="mean",
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in self.input_dims
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.topic_transform = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.article_bias = nn.Parameter(torch.zeros(1))
        self.topic_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.topic_transform)

    def encode(
        self,
        features: Mapping[str, Tensor],
        edge_indices: Mapping[EdgeType, Tensor],
    ) -> dict[str, Tensor]:
        """Generate contextual node embeddings with PyG ``HeteroConv``."""
        hidden = {
            node_type: F.relu(self.projections[node_type](features[node_type]))
            for node_type in self.input_dims
        }
        for conv, norms in zip(self.convs, self.norms, strict=True):
            updates = conv(hidden, edge_indices)
            hidden = {
                node_type: F.dropout(
                    norms[node_type](updates.get(node_type, value) + value),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type, value in hidden.items()
            }
        return hidden

    def forward(
        self,
        features: Mapping[str, Tensor],
        edge_indices: Mapping[EdgeType, Tensor],
    ) -> Tensor:
        """Return logits shaped ``(article_count, topic_count)``."""
        hidden = self.encode(features, edge_indices)
        return (
            hidden["article"]
            @ self.topic_transform
            @ hidden["topic"].transpose(0, 1)
            + self.article_bias
            + self.topic_bias
        )

    def save(self, path: str) -> None:
        """Save model structure and weights using friend baseline format."""
        torch.save(
            {
                "model_variant": "pyg_hetero_graphsage",
                "config": {
                    "input_dims": self.input_dims,
                    "edge_types": [list(edge_type) for edge_type in self.edge_types],
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
                "weights": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> HeterogeneousGraphSAGE:
        """Load weights and graph metadata from a checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(checkpoint["config"])
        config["edge_types"] = [tuple(value) for value in config["edge_types"]]
        model = cls(**config)
        model.load_state_dict(checkpoint["weights"])
        model.eval()
        return model
