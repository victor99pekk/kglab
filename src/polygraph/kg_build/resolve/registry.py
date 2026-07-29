"""Lazy registry for entity-resolution methods."""

from importlib import import_module

RESOLUTION_METHODS = {
    "embedding": "polygraph.kg_build.resolve.embedding:resolve_embedding",
    "string": "polygraph.kg_build.resolve.string:resolve_string",
}


def get_resolution_method(method: str):
    if method == "string_similarity":
        method = "string"
    try:
        import_path = RESOLUTION_METHODS[method]
    except KeyError as exc:
        choices = ", ".join(sorted(RESOLUTION_METHODS))
        raise ValueError(f"Unknown resolution method '{method}'. Available: {choices}") from exc
    module_name, function_name = import_path.split(":", maxsplit=1)
    return getattr(import_module(module_name), function_name)


__all__ = ["RESOLUTION_METHODS", "get_resolution_method"]
