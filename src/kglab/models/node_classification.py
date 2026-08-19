"""Node classification inference tool — loads a trained GNN and predicts entity types.

This is the **inference-only** counterpart to the training code in
``src/ml/node_classification/``. Pipelines import this, not the training code.

Usage:
    from kglab.models.node_classification import NodeClassificationTool

    tool = NodeClassificationTool()
    tool.load("experiments/ML_models/001_nc/models/best.pt")
    entity_type = tool.predict(entity_dict, graph)
"""

from __future__ import annotations

from typing import Any


class NodeClassificationTool:
    """Inference-only wrapper around a trained node classification GNN.

    Load a trained checkpoint and call ``predict()`` to get the entity type
    for a node, or ``predict_all()`` to classify all nodes in a graph.

    Attributes:
        model: The underlying GNN model (set by ``load()``).
        label_names: Dict mapping class index → type name (e.g. 0 → "PERSON").
        is_loaded: Whether a checkpoint has been loaded.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.label_names: dict[int, str] = {}

    def load(self, path: str) -> None:
        """Load a trained model checkpoint from disk.

        Args:
            path: Path to the saved checkpoint file.

        Raises:
            NotImplementedError: Override in a real PyTorch subclass.
        """
        # TODO: Replace with actual checkpoint loading:
        #   import torch
        #   from ml.node_classification.models import get_model
        #   ckpt = torch.load(path)
        #   model_cls = get_model(ckpt["model_variant"])
        #   self.model = model_cls(**ckpt["config"])
        #   self.model.load_state_dict(ckpt["weights"])
        #   self.label_names = ckpt["label_names"]
        raise NotImplementedError(
            "Override NodeClassificationTool.load() with real checkpoint loading."
        )

    def predict(self, entity: dict[str, Any], graph: Any) -> str:
        """Predict the type of a single entity node.

        Args:
            entity: Entity dict with features.
            graph: The full graph (for neighborhood context).

        Returns:
            Predicted type string (e.g. "PERSON", "ORG").

        Raises:
            RuntimeError: If no model has been loaded yet.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load(path) first.")
        raise NotImplementedError(
            "Override NodeClassificationTool.predict() with real inference logic."
        )

    def predict_all(self, graph: Any) -> dict[str, str]:
        """Predict types for all entity nodes in a graph.

        Returns:
            Dict mapping entity ID → predicted type string.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load(path) first.")
        raise NotImplementedError(
            "Override NodeClassificationTool.predict_all() with real inference logic."
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Auto-register with the ModelRegistry on import
from kglab.models.registry import ModelRegistry  # noqa: E402

ModelRegistry.register("node_classification", NodeClassificationTool)
