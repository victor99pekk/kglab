"""MLP entity resolution model — binary classifier on entity feature vectors.

Architecture: takes two entity feature vectors, computes element-wise
difference and product, concatenates them, and passes through a multi-layer
perceptron to a sigmoid binary output.

This implements the ``BaseEntityResolver`` protocol so it can be used via
the model registry in ``ml.entity_resolution.models``.
"""

from __future__ import annotations

from typing import Any


class MLPEntityResolver:
    """Binary MLP classifier for entity mention pairs.

    Args:
        input_dim: Dimension of each entity's feature vector.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout probability between hidden layers.
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
            "Replace with a real PyTorch nn.Module or sklearn Pipeline. "
            "The architecture: diff = |a - b|, prod = a * b, "
            "concat(a, b, diff, prod) → MLP → sigmoid."
        )

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        raise NotImplementedError("Override save() in a real PyTorch subclass.")

    @classmethod
    def load(cls, path: str) -> MLPEntityResolver:
        """Load model weights from disk."""
        raise NotImplementedError("Override load() in a real PyTorch subclass.")

    @property
    def is_trained(self) -> bool:
        return self._trained
