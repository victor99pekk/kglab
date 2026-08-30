"""Configuration for node classification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NodeClassificationConfig:
    """Hyperparameters and settings for node classification training.

    Attributes:
        model_variant: Which GNN architecture to use ("gcn", "gat", ...).
        embedding_dim: Dimension of entity feature vectors.
        hidden_dim: GNN hidden layer dimension.
        num_layers: Number of GNN convolution layers.
        dropout: Dropout rate.
        min_entity_frequency: Skip entity types with fewer than this many examples.
        train_split: Fraction of nodes to use for training.
        val_split: Fraction of nodes to use for validation (from train remainder).
    """

    model_variant: str = "gcn"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    min_entity_frequency: int = 2
    train_split: float = 0.8
    val_split: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeClassificationConfig:
        """Create a config from a flat dict (e.g., parsed from experiment YAML)."""
        return cls(
            model_variant=data.get("variant", "gcn"),
            embedding_dim=data.get("embedding_dim", 128),
            hidden_dim=data.get("hidden_dim", 256),
            num_layers=data.get("num_layers", 2),
            dropout=data.get("dropout", 0.3),
            min_entity_frequency=data.get("min_entity_frequency", 2),
            train_split=data.get("train_split", 0.8),
            val_split=data.get("val_split", 0.1),
        )
