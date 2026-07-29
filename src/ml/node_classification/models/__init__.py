"""Node classification model registry — one entry per GNN architecture.

Models are lazy-imported so that PyTorch Geometric is only required when
a model is actually instantiated, not when the package is imported.

Usage:
    from ml.node_classification.models import MODEL_REGISTRY, get_model

    model_cls = get_model("gat")
    model = model_cls(input_dim=128, hidden_dim=256, num_classes=9, num_heads=4)
"""

from __future__ import annotations

import importlib

#: Maps variant name strings → (module_path, class_name) tuples.
#: Models are lazy-imported on first access via ``get_model()``.
#: Add new entries here::
#:
#:     MODEL_REGISTRY["new_variant"] = ("ml.node_classification.models.new_variant", "NewModel")
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gcn": ("ml.node_classification.models.gcn", "GCNNodeClassifier"),
    "gat": ("ml.node_classification.models.gat", "GATNodeClassifier"),
}

#: Cache of already-imported model classes.
_IMPORT_CACHE: dict[str, type] = {}


def get_model(variant: str) -> type:
    """Return the model class for a given variant name (lazy import).

    The model is imported on first access and cached for subsequent calls.
    """
    if variant not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown node classification model variant: '{variant}'. Available: {available}"
        )

    if variant not in _IMPORT_CACHE:
        mod_path, cls_name = MODEL_REGISTRY[variant]
        module = importlib.import_module(mod_path)
        _IMPORT_CACHE[variant] = getattr(module, cls_name)

    return _IMPORT_CACHE[variant]


def available_variants() -> list[str]:
    """Return the list of registered model variant names."""
    return list(MODEL_REGISTRY.keys())
