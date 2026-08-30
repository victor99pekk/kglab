"""Access to the benchmark gold datasets bundled with the library.

The small gold files used as each stage benchmark's default live *inside*
the installed ``kglab`` package (``kglab.benchmark_pipeline.data``) and are
shipped in wheels via ``[tool.setuptools.package-data]``.  Resolving them
through :mod:`importlib.resources` — never through a cwd-relative path —
means benchmarks work from any working directory, in editable installs,
and inside installed wheels.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

#: Package-relative location of the bundled gold datasets.
_BUNDLED_GOLD_DIR = "data"


def bundled_data_dir() -> Path:
    """Absolute path to the benchmark gold datasets shipped with ``kglab``.

    Returns the directory holding the ``*_gold.jsonl`` files (dedup,
    chunking, ner, ner_test, ner_wikiann, quality, resolution) that the
    stage benchmarks score by default.  Each ``Benchmark.<Stage>`` falls
    back to its bundled file when no ``dataset=`` is given.

    Example::

        from kglab.benchmark_pipeline import bundled_data_dir

        gold = bundled_data_dir() / "ner_gold.jsonl"
    """
    return Path(resources.files("kglab.benchmark_pipeline") / _BUNDLED_GOLD_DIR)
