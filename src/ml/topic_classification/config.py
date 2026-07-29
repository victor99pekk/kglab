"""Configuration contract for Level-1 topic-classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TopicClassificationConfig:
    """Model and data choices shared by topic-classification experiments."""

    model_variant: str = "pyg_hetero_graphsage"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    train_split: float = 0.7
    val_split: float = 0.15
    threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicClassificationConfig:
        """Create configuration from an experiment YAML model section."""
        return cls(
            model_variant=str(data.get("variant", "pyg_hetero_graphsage")),
            hidden_dim=int(data.get("hidden_dim", 128)),
            num_layers=int(data.get("num_layers", 2)),
            dropout=float(data.get("dropout", 0.2)),
            train_split=float(data.get("train_split", 0.7)),
            val_split=float(data.get("val_split", 0.15)),
            threshold=float(data.get("threshold", 0.5)),
        )
