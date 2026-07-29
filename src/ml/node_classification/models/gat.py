"""GAT node classifier — predicts entity type using graph attention.

Architecture: stacked GATConv layers with multi-head attention and ELU
activation, followed by a linear classification head. Attention weights
let the model focus on relevant neighbors for each entity.

Uses PyTorch Geometric. Install with:
    pip install torch-geometric
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — PyTorch convention
from torch_geometric.nn import GATConv


class GATNodeClassifier(nn.Module):
    """GAT-based node classifier for entity type prediction.

    Args:
        input_dim: Dimension of entity feature vectors.
        hidden_dim: GAT hidden layer dimension (per head).
        num_classes: Number of output classes (entity types from ontology).
        num_layers: Number of GAT convolution layers (≥ 2).
        num_heads: Number of attention heads per layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 9,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim
            heads = num_heads if i < num_layers - 1 else 1
            concat = i < num_layers - 1

            self.convs.append(GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout))
            norm_in = out_dim * heads if concat else out_dim
            self.norms.append(nn.BatchNorm1d(norm_in))

        # After the last GATConv (concat=False), output is hidden_dim * 1
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute class logits for all nodes with multi-head attention.

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
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def save(self, path: str) -> None:
        """Save model weights and config to disk."""
        torch.save(
            {
                "model_variant": "gat",
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_classes": self.num_classes,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                },
                "weights": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> GATNodeClassifier:
        """Load model weights and config from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["weights"])
        model.eval()
        return model
