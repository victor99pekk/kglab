"""Lazy registry for graph-construction methods."""

from importlib import import_module
from typing import Any

BUILD_METHODS = {
    "networkx": "polygraph.kg_build.build.networkx:GraphBuilder",
    "sqlite": "polygraph.kg_build.build.sqlite:SQLiteGraphBuilder",
}


def create_build_method(method: str, **kwargs: Any):
    try:
        import_path = BUILD_METHODS[method]
    except KeyError as exc:
        choices = ", ".join(sorted(BUILD_METHODS))
        raise ValueError(f"Unknown graph build method '{method}'. Available: {choices}") from exc
    module_name, class_name = import_path.split(":", maxsplit=1)
    method_type = getattr(import_module(module_name), class_name)
    return method_type(**kwargs)


__all__ = ["BUILD_METHODS", "create_build_method"]
