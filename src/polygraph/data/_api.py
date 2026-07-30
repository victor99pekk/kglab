"""Public API for dataset download and enrichment.

.. code-block:: python

    from polygraph.data import Data

    Data.list()
    Data.download("wikipedia_random", path="data/wikipedia/", count=100, enrich=True)
"""

from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Data:
    """Download and enrich supported datasets for Polygraph pipelines.

    All methods are static.  Supported datasets are registered in
    ``DATASET_REGISTRY`` — each entry maps a short name to a description,
    a download function, and an optional enrich function.

    Adding support for a new dataset means:

    1. Write a module under ``polygraph.data/`` with a ``download_*``
       (and optionally ``enrich_*``) function.
    2. Add an entry to ``DATASET_REGISTRY`` in ``polygraph.data.__init__``.
    """

    @staticmethod
    def list() -> list[dict[str, str]]:
        """Return all supported datasets with name and description.

        Returns:
            List of ``{"name": ..., "description": ...}`` dicts.
        """
        from polygraph.data import DATASET_REGISTRY

        return [
            {"name": name, "description": info["description"]}
            for name, info in DATASET_REGISTRY.items()
        ]

    @staticmethod
    def info(name: str) -> dict[str, Any]:
        """Return metadata for a supported dataset.

        Args:
            name: Dataset short name (e.g., ``"wikipedia_random"``).

        Returns:
            Dict with ``name``, ``description``, ``has_enrich``, ``has_download``.
        """

        entry = _resolve(name)
        return {
            "name": name,
            "description": entry["description"],
            "has_download": entry.get("download") is not None,
            "has_enrich": entry.get("enrich") is not None,
        }

    @staticmethod
    def download(
        name: str,
        path: str | Path,
        enrich: bool = False,
        force: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Download (and optionally enrich) a supported dataset.

        By default, skips download if *path* already exists and is non-empty.
        Set ``force=True`` to always re-download.

        Args:
            name: Dataset short name (e.g., ``"wikipedia_random"``).
            path: Output path for the downloaded JSONL file.
            enrich: If ``True``, run the enrichment step after download.
            force: If ``True``, re-download even if *path* already has data.
            **kwargs: Passed to the dataset-specific download function
                      (e.g., ``count=100`` for ``wikipedia_random"``).

        Returns:
            Dict with ``dataset``, ``path``, ``cached`` (``True`` if skipped),
            ``downloaded`` (record count, if not cached), and ``enriched``
            (record count, only when ``enrich=True``).

        Raises:
            ValueError: If the dataset name is unknown.
        """

        entry = _resolve(name)
        result: dict[str, Any] = {"dataset": name, "path": str(path)}

        # Cache check: skip if file already has data (unless forced)
        target = Path(path)
        if not force and _file_has_records(target):
            result["cached"] = True
            result["downloaded"] = 0
            logger.info(
                "Skipping download — %s already has data (use force=True to re-download)", target
            )
        else:
            download_fn = _import_fn(entry["download"])
            count = download_fn(path=path, **kwargs)
            result["downloaded"] = count
            result["cached"] = False

        if enrich and entry.get("enrich"):
            enrich_fn = _import_fn(entry["enrich"])
            # Only pass kwargs that the enrich function accepts
            enrich_kwargs: dict[str, Any] = {}
            for key in ("language", "delay"):
                if key in kwargs:
                    enrich_kwargs[key] = kwargs[key]
            enriched = enrich_fn(input_path=path, force=force, **enrich_kwargs)
            result["enriched"] = enriched
        elif enrich and not entry.get("enrich"):
            raise ValueError(f"Dataset '{name}' does not support enrichment.")

        return result

    @staticmethod
    def enrich(
        name: str,
        input_path: str | Path,
        output_path: str | Path | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> int:
        """Enrich an already-downloaded dataset.

        By default, skips enrichment if records already contain the expected
        enrichment fields (e.g., ``links``).  Set ``force=True`` to always
        re-enrich.

        Args:
            name: Dataset short name (e.g., ``"wikipedia_random"``).
            input_path: Path to the existing JSONL file.
            output_path: Where to write enriched output (defaults to overwriting input).
            force: If ``True``, re-enrich even if records already have links.
            **kwargs: Passed to the dataset-specific enrich function.

        Returns:
            Number of records enriched (0 if cached).

        Raises:
            ValueError: If the dataset name is unknown or has no enrich step.
        """

        entry = _resolve(name)
        if not entry.get("enrich"):
            raise ValueError(f"Dataset '{name}' does not support enrichment.")
        enrich_fn = _import_fn(entry["enrich"])
        return enrich_fn(input_path=input_path, output_path=output_path, force=force, **kwargs)


def _resolve(name: str) -> dict[str, Any]:
    """Look up a dataset in the registry."""
    from polygraph.data import DATASET_REGISTRY

    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}") from None


def _import_fn(import_path: str) -> Any:
    """Lazy-import a function from a ``"module.path:function_name"`` string."""
    module_name, function_name = import_path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, function_name)


def _file_has_records(path: Path) -> bool:
    """Return ``True`` if *path* exists and contains at least one valid JSONL record."""
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)  # validate at least one line is parseable JSON
                return True
        return False
    except (json.JSONDecodeError, OSError):
        return False
