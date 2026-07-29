"""GCN node classifier — predicts entity type from graph neighborhood.

Architecture: stacked GCNConv layers with ReLU activation and dropout,
followed by a linear classification head.

Uses PyTorch Geometric. Install with:
    pip install torch-geometric
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — PyTorch convention
from torch_geometric.nn import GCNConv


class GCNNodeClassifier(nn.Module):
    """GCN-based node classifier for entity type prediction.

    Args:
        input_dim: Dimension of entity feature vectors.
        hidden_dim: GCN hidden layer dimension.
        num_classes: Number of output classes (entity types from ontology).
        num_layers: Number of GCN convolution layers (≥ 2).
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 9,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute class logits for all nodes.

        Args:
            features: ``(num_nodes, input_dim)`` tensor of entity features.
            edge_index: ``(2, num_edges)`` tensor of graph connectivity.

        Returns:
            ``(num_nodes, num_classes)`` tensor of class logits.
        """
        x = features

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def save(self, path: str) -> None:
        """Save model weights and config to disk."""
        torch.save(
            {
                "model_variant": "gcn",
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_classes": self.num_classes,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
                "weights": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> GCNNodeClassifier:
        """Load model weights and config from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["weights"])
        model.eval()
        return model
