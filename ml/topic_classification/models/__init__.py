"""Lazy model registry for topic-classification architectures."""

from __future__ import annotations

import importlib

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "pyg_hetero_graphsage": (
        "ml.topic_classification.models.hetero_graphsage",
        "HeterogeneousGraphSAGE",
    ),
}

_IMPORT_CACHE: dict[str, type] = {}


def get_model(variant: str) -> type:
    """Return registered model class, importing PyG only when requested."""
    if variant not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown topic model variant: '{variant}'. Available: {available}")
    if variant not in _IMPORT_CACHE:
        module_path, class_name = MODEL_REGISTRY[variant]
        module = importlib.import_module(module_path)
        _IMPORT_CACHE[variant] = getattr(module, class_name)
    return _IMPORT_CACHE[variant]


def available_variants() -> list[str]:
    """Return registered topic-classification architecture names."""
    return list(MODEL_REGISTRY)


__all__ = [
    "MODEL_REGISTRY",
    "available_variants",
    "get_model",
]
