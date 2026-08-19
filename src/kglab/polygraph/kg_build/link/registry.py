"""Lazy registry for entity-linking methods."""

from __future__ import annotations

from importlib import import_module
from typing import Any

LINKER_REGISTRY: dict[str, str] = {}


def create_linker(method: str, **kwargs: Any):
    """Instantiate a linker by name from the registry.

    Args:
        method: Registered name (e.g. ``"wikidata"``).
        **kwargs: Forwarded to the linker constructor.

    Returns:
        An ``EntityLinker`` instance.

    Raises:
        ValueError: If *method* is not in the registry.
    """
    try:
        import_path = LINKER_REGISTRY[method]
    except KeyError as exc:
        choices = ", ".join(sorted(LINKER_REGISTRY))
        raise ValueError(f"Unknown linker '{method}'. Available: {choices}") from exc
    module_name, class_name = import_path.split(":", maxsplit=1)
    linker_type = getattr(import_module(module_name), class_name)
    return linker_type(**kwargs)


__all__ = ["LINKER_REGISTRY", "create_linker"]
