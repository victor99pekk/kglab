"""KGLab — convenience alias for the `polygraph` package.

This module re-exports the main `polygraph` submodules under the
`kglab` top-level name so that both `import polygraph` and
`import kglab` work while a full rename is staged. It is intentionally
minimal and imports lazily at module import time.
"""

from importlib import import_module

__all__ = [
    "__version__",
]

try:
    _root = import_module("polygraph")
    __version__ = getattr(_root, "__version__", "0.0.0")
except Exception:
    __version__ = "0.0.0"

# Re-export common submodules so `from kglab import pipelines` works.
_submodules = [
    "pipelines",
    "benchmark_pipeline",
    "models",
    "preprocess",
    "kg_build",
    "kg_eval",
    "kg_export",
    "finetune",
    "data",
    "_shared",
]

for _sub in _submodules:
    try:
        _m = import_module(f"polygraph.{_sub}")
        globals()[_sub] = _m
        __all__.append(_sub)
    except Exception:
        # best-effort: if a submodule is missing, skip it silently.
        continue
