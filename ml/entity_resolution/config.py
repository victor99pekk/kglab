"""Configuration for entity resolution training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityResolutionConfig:
    """Hyperparameters and settings for entity resolution training.

    Attributes:
        model_variant: Which architecture to use ("mlp", "attention", ...).
        embedding_dim: Dimension of entity feature vectors.
        hidden_dims: Hidden layer sizes for the classifier head.
        dropout: Dropout rate between hidden layers.
        num_heads: Number of attention heads (attention variant only).
        min_entity_frequency: Entities mentioned fewer times than this are excluded.
        negative_ratio: Number of negative pairs to generate per positive pair.
        train_split: Fraction of entity pairs to use for training.
        val_split: Fraction of entity pairs to use for validation (from train remainder).
    """

    model_variant: str = "mlp"
    embedding_dim: int = 128
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3
    num_heads: int = 4
    min_entity_frequency: int = 2
    negative_ratio: int = 3
    train_split: float = 0.8
    val_split: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntityResolutionConfig:
        """Create a config from a flat dict (e.g., parsed from experiment YAML)."""
        return cls(
            model_variant=data.get("variant", "mlp"),
            embedding_dim=data.get("embedding_dim", 128),
            hidden_dims=data.get("hidden_dims", [256, 128]),
            dropout=data.get("dropout", 0.3),
            num_heads=data.get("num_heads", 4),
            min_entity_frequency=data.get("min_entity_frequency", 2),
            negative_ratio=data.get("negative_ratio", 3),
            train_split=data.get("train_split", 0.8),
            val_split=data.get("val_split", 0.1),
        )
