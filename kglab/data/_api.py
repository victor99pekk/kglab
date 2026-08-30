"""Public API for dataset download and enrichment.

.. code-block:: python

    from kglab.data import Data, RandomSampler

    Data.list()
    Data.download("wikipedia", path="data/wikipedia/", sampler=RandomSampler(count=100))
"""

from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Data:
    """Download and enrich supported datasets for KGLab pipelines.

    All methods are static.  Supported datasets are registered in
    ``DATASET_REGISTRY`` — each entry maps a short name to a description,
    a download function, and an optional enrich function.

    Adding support for a new dataset means:

    1. Write a module under ``kglab.data/`` with a ``download_*``
       (and optionally ``enrich_*``) function.
    2. Add an entry to ``DATASET_REGISTRY`` in ``kglab.data.__init__``.
    """

    @staticmethod
    def list() -> list[dict[str, str]]:
        """Return all supported datasets with name and description.

        Returns:
            List of ``{"name": ..., "description": ...}`` dicts.
        """
        from kglab.data import DATASET_REGISTRY

        return [
            {"name": name, "description": info["description"]}
            for name, info in DATASET_REGISTRY.items()
        ]

    @staticmethod
    def info(name: str) -> dict[str, Any]:
        """Return metadata for a supported dataset.

        Args:
            name: Dataset short name (e.g., ``"wikipedia"``).

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
        path: str | Path | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Download a supported dataset and always run its enrichment step.

        By default, skips download if the target path already exists and is
        non-empty.  Set ``force=True`` to always re-download.  When the
        dataset supports enrichment (e.g., Wikipedia hyperlinks), it is run
        after every call — enrichment itself is cached and skips records that
        already contain the enrichment fields.

        Args:
            name: Dataset short name (e.g., ``"wikipedia"``).
            path: Output path for the downloaded JSONL file.  Defaults to
                ``data/{name}.jsonl``.
            force: If ``True``, re-download even if the file already has data.
            **kwargs: Passed to the dataset-specific download function
                      (e.g., ``sampler=RandomSampler(count=100)`` for
                      ``"wikipedia"``).  ``language`` and ``delay`` are also
                      forwarded to the enrichment step when supported.

        Returns:
            Dict with ``dataset``, ``path``, ``cached`` (``True`` if skipped),
            and ``downloaded`` (record count, if not cached).  Enrichment
            still runs whenever the dataset supports it, but its count is
            not included in the returned dict.

        Raises:
            ValueError: If the dataset name is unknown.
        """

        entry = _resolve(name)
        target = Path(path) if path is not None else Path("data") / f"{name}.jsonl"
        result: dict[str, Any] = {"dataset": name, "path": str(target)}

        # Cache check: skip if file already has data (unless forced)
        if not force and _file_has_records(target):
            result["cached"] = True
            result["downloaded"] = 0
            logger.info(
                "Skipping download — %s already has data (use force=True to re-download)", target
            )
        else:
            download_fn = _import_fn(entry["download"])
            count = download_fn(path=target, **kwargs)
            result["downloaded"] = count
            result["cached"] = False

        # Enrichment always runs when the dataset supports it (cached internally).
        if entry.get("enrich"):
            enrich_fn = _import_fn(entry["enrich"])
            # Only pass kwargs that the enrich function accepts
            enrich_kwargs: dict[str, Any] = {}
            for key in ("language", "delay"):
                if key in kwargs:
                    enrich_kwargs[key] = kwargs[key]
            enrich_fn(input_path=target, force=force, **enrich_kwargs)

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
            name: Dataset short name (e.g., ``"wikipedia"``).
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
    from kglab.data import DATASET_REGISTRY

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
