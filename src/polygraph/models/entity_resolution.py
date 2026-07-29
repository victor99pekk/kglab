"""Entity resolution inference tool — loads a trained checkpoint and resolves pairs.

This is the **inference-only** counterpart to the training code in
``src/ml/entity_resolution/``. Pipelines import this, not the training code.

Usage:
    from polygraph.models.entity_resolution import EntityResolutionTool

    tool = EntityResolutionTool()
    tool.load("experiments/ML_models/001_er/models/best.pt")
    is_match = tool.resolve(entity_a_dict, entity_b_dict)
"""

from __future__ import annotations

from typing import Any


class EntityResolutionTool:
    """Inference-only wrapper around a trained entity resolution model.

    Load a trained checkpoint and call ``resolve()`` to predict whether
    two entity mentions refer to the same real-world entity.

    Attributes:
        model: The underlying model (set by ``load()``).
        is_loaded: Whether a checkpoint has been loaded.
    """

    def __init__(self) -> None:
        self.model: Any = None

    def load(self, path: str) -> None:
        """Load a trained model checkpoint from disk.

        Args:
            path: Path to the saved checkpoint file.

        Raises:
            NotImplementedError: Override in a real PyTorch/sklearn subclass.
        """
        # TODO: Replace with actual checkpoint loading:
        #   import torch
        #   from ml.entity_resolution.model import EntityResolutionModel
        #   self.model = EntityResolutionModel(...)
        #   self.model.load_state_dict(torch.load(path))
        raise NotImplementedError(
            "Override EntityResolutionTool.load() with real checkpoint loading "
            "(e.g., torch.load, joblib.load). The training code lives in "
            "src/ml/entity_resolution/."
        )

    def resolve(self, entity_a: dict[str, Any], entity_b: dict[str, Any]) -> bool:
        """Predict whether two entity mentions are the same entity.

        Args:
            entity_a: Dict with at least a ``"name"`` key.
            entity_b: Dict with at least a ``"name"`` key.

        Returns:
            ``True`` if the entities are predicted to match.

        Raises:
            RuntimeError: If no model has been loaded yet.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load(path) first.")
        # TODO: Replace with actual inference:
        #   features_a = self._encode(entity_a)
        #   features_b = self._encode(entity_b)
        #   prob = self.model.forward(features_a, features_b)
        #   return prob > 0.5
        raise NotImplementedError(
            "Override EntityResolutionTool.resolve() with real inference logic."
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Auto-register with the ModelRegistry on import
from polygraph.models.registry import ModelRegistry  # noqa: E402

ModelRegistry.register("entity_resolution", EntityResolutionTool)
