"""Entity resolution model — binary classifier on entity feature vectors.

Architecture: a simple MLP that takes two entity feature vectors, computes
element-wise difference and product, concatenates them, and passes through
hidden layers to a binary output.

This is a reference implementation. Replace with a GNN, transformer, or
any other architecture by subclassing and overriding ``forward()``.
"""

from __future__ import annotations

from typing import Any


class EntityResolutionModel:
    """Binary classifier for entity mention pairs.

    This is a framework-agnostic stub. In production, replace with a
    PyTorch ``nn.Module``, sklearn ``Pipeline``, or any trainable model.

    The model implements a ``save(path)`` method for checkpointing (required
    by ``SaveBest`` from ``polygraph.models.training_utils``).

    Args:
        input_dim: Dimension of each entity's feature vector.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout probability between layers.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self._trained = False

    def forward(self, features_a: Any, features_b: Any) -> Any:
        """Compute match probability for a batch of entity pairs.

        Args:
            features_a: (batch_size, input_dim) tensor of entity A features.
            features_b: (batch_size, input_dim) tensor of entity B features.

        Returns:
            (batch_size, 1) tensor of match probabilities in [0, 1].
        """
        raise NotImplementedError(
            "Subclass EntityResolutionModel and implement forward(), or use "
            "a PyTorch nn.Module / sklearn Pipeline instead."
        )

    def save(self, path: str) -> None:
        """Save model weights to disk (stub — override in subclass)."""
        raise NotImplementedError("Override save() in your model subclass.")

    @classmethod
    def load(cls, path: str) -> EntityResolutionModel:
        """Load model weights from disk (stub — override in subclass)."""
        raise NotImplementedError("Override load() in your model subclass.")

    @property
    def is_trained(self) -> bool:
        return self._trained
