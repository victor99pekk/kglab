"""Entity resolution model registry — one entry per architecture variant.

Usage:
    from ml.entity_resolution.models import MODEL_REGISTRY, get_model

    model_cls = get_model("attention")
    model = model_cls(input_dim=128, num_heads=4)
"""

from __future__ import annotations

from ml.entity_resolution.models.attention import AttentionEntityResolver
from ml.entity_resolution.models.mlp import MLPEntityResolver

#: Maps variant name strings → model classes.
#: Add new entries here when creating a new architecture::
#:
#:     from ml.entity_resolution.models.gnn import GNNEntityResolver
#:     MODEL_REGISTRY["gnn"] = GNNEntityResolver
MODEL_REGISTRY: dict[str, type] = {
    "mlp": MLPEntityResolver,
    "attention": AttentionEntityResolver,
}


def get_model(variant: str) -> type:
    """Return the model class for a given variant name.

    Args:
        variant: One of the keys in ``MODEL_REGISTRY``.

    Returns:
        The model class (not instantiated).

    Raises:
        KeyError: If *variant* is not registered.
    """
    if variant not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown entity resolution model variant: '{variant}'. Available: {available}"
        )
    return MODEL_REGISTRY[variant]


def available_variants() -> list[str]:
    """Return the list of registered model variant names."""
    return list(MODEL_REGISTRY.keys())


def model_config_keys(variant: str) -> list[str]:
    """Return the constructor parameter names for a model variant.

    Useful for validating that experiment YAML configs pass the right
    keyword arguments to the model's ``__init__``.
    """
    import inspect

    model_cls = get_model(variant)
    params = inspect.signature(model_cls.__init__).parameters
    return [p for p in params if p != "self"]
