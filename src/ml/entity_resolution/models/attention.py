"""Attention-based entity resolution model.

Architecture: uses cross-attention between two entity mention contexts
(followed by a pooling and MLP head) to predict whether two mentions
refer to the same real-world entity.

This is a **skeleton** — replace with a real PyTorch implementation
using ``torch.nn.MultiheadAttention`` or a small transformer encoder.
"""

from __future__ import annotations

from typing import Any


class AttentionEntityResolver:
    """Cross-attention classifier for entity mention pairs.

    Args:
        input_dim: Dimension of entity feature vectors (token-level).
        num_heads: Number of attention heads.
        hidden_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self._trained = False

    def forward(self, features_a: Any, features_b: Any) -> Any:
        """Compute match probability via cross-attention.

        Args:
            features_a: (batch_size, seq_len, input_dim) token-level features for entity A.
            features_b: (batch_size, seq_len, input_dim) token-level features for entity B.

        Returns:
            (batch_size, 1) tensor of match probabilities in [0, 1].
        """
        raise NotImplementedError(
            "Replace with real PyTorch: cross-attention(a, b) → pool → MLP → sigmoid."
        )

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        raise NotImplementedError("Override save() in a real PyTorch subclass.")

    @classmethod
    def load(cls, path: str) -> AttentionEntityResolver:
        """Load model weights from disk."""
        raise NotImplementedError("Override load() in a real PyTorch subclass.")

    @property
    def is_trained(self) -> bool:
        return self._trained
