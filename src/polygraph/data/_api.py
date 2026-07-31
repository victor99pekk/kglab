"""Public API for dataset download, enrichment, and merging.

.. code-block:: python

    from polygraph.data import Data

    Data.list()
    Data.download("wikipedia_random", path="data/wikipedia/", count=100, enrich=True)
    Data.merge(["part1.jsonl", "part2.jsonl"], output="merged.jsonl", dedup_key="id")
    Data.download_many([
        {"name": "wikipedia_random", "count": 50, "language": "en"},
        {"name": "wikipedia_random", "count": 30, "language": "fr"},
    ], output="merged.jsonl", dedup_key="id")
"""

from __future__ import annotations

import contextlib
import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Data:
    """Download, enrich, and merge supported datasets for Polygraph pipelines.

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

    @staticmethod
    def merge(
        paths: list[str | Path],
        output: str | Path,
        dedup_key: str | None = None,
        keep: str = "first",
    ) -> dict[str, Any]:
        """Merge multiple JSONL files into one, optionally deduplicating.

        Without deduplication the merge is a streaming concatenation
        (constant memory).  With ``dedup_key`` set, all records are read into
        memory so that ``keep="last"`` works correctly — keep this in mind for
        very large datasets.

        Args:
            paths: List of JSONL file paths to merge.
            output: Path for the merged output file.
            dedup_key: If set, deduplicate records by this JSON field.
            keep: When deduplicating, keep ``"first"`` or ``"last"``
                  occurrence of each key.

        Returns:
            Dict with ``merged`` (total records written), ``duplicates_removed``,
            and ``output``.

        Raises:
            FileNotFoundError: If any input path does not exist.
            ValueError: If *keep* is not ``"first"`` or ``"last"``,
                        or a record is missing the *dedup_key* field.
        """
        if keep not in ("first", "last"):
            raise ValueError(f"keep must be 'first' or 'last', got '{keep}'")

        target = Path(output)

        if dedup_key is None:
            # Streaming concatenation — no dedup needed
            return _merge_streaming(paths, target)

        # In-memory merge with deduplication
        return _merge_dedup(paths, target, dedup_key, keep)

    @staticmethod
    def download_many(
        specs: list[dict[str, Any]],
        output: str | Path,
        enrich: bool = False,
        force: bool = False,
        dedup_key: str | None = None,
        keep: str = "first",
    ) -> dict[str, Any]:
        """Download multiple datasets and merge into one output file.

        Each spec is a dict with a ``"name"`` key (dataset short name) plus
        any keyword arguments for that dataset's download function.  Temporary
        files are created for intermediate downloads and cleaned up after merging.

        Args:
            specs: List of download specs. Each must contain ``"name"`` plus
                   dataset-specific kwargs (e.g., ``count``, ``language``).
            output: Path for the merged output file.
            enrich: Passed through to each ``Data.download()`` call.
            force: Passed through to each ``Data.download()`` call.
            dedup_key: If set, deduplicate merged records by this JSON field.
            keep: When deduplicating, keep ``"first"`` or ``"last"`` occurrence.

        Returns:
            Merge result dict (see ``Data.merge()``) with an additional
            ``downloads`` key listing each individual download result.

        Raises:
            ValueError: If any spec is missing a ``"name"`` key.
        """
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="polygraph_dl_"))
        temp_paths: list[Path] = []
        download_results: list[dict[str, Any]] = []

        try:
            for i, spec in enumerate(specs):
                name = spec.get("name")
                if name is None:
                    raise ValueError(f"Spec {i} is missing required 'name' key: {spec}")
                # Build kwargs without mutating the caller's dict
                download_kwargs = {k: v for k, v in spec.items() if k != "name"}
                tmp_path = temp_dir / f"part_{i:03d}.jsonl"
                temp_paths.append(tmp_path)

                result = Data.download(
                    name=name,
                    path=str(tmp_path),
                    enrich=enrich,
                    force=force,
                    **download_kwargs,
                )
                download_results.append(result)
                logger.info(
                    "Downloaded %s → %s (%d records)",
                    name,
                    tmp_path,
                    result.get("downloaded", 0),
                )

            merge_result = Data.merge(
                paths=[str(p) for p in temp_paths],
                output=output,
                dedup_key=dedup_key,
                keep=keep,
            )
            merge_result["downloads"] = download_results
            return merge_result

        finally:
            # Clean up temp files
            for p in temp_paths:
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                temp_dir.rmdir()


def _merge_streaming(paths: list[str | Path], target: Path) -> dict[str, Any]:
    """Concatenate JSONL files without deduplication (constant memory)."""
    total = 0

    with target.open("w", encoding="utf-8") as out:
        for p in paths:
            source = Path(p)
            if not source.exists():
                raise FileNotFoundError(f"Merge input not found: {source}")
            with source.open(encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    out.write(line if line.endswith("\n") else line + "\n")
                    total += 1

    logger.info("Merged %d records into %s (no dedup)", total, target)
    return {"merged": total, "duplicates_removed": 0, "output": str(target)}


def _merge_dedup(
    paths: list[str | Path],
    target: Path,
    dedup_key: str,
    keep: str,
) -> dict[str, Any]:
    """Merge JSONL files with deduplication (reads all records into memory)."""
    # key_str → (line, source_index) — source_index preserves input order
    unique: dict[str, tuple[str, int]] = {}
    duplicates = 0

    for i, p in enumerate(paths):
        source = Path(p)
        if not source.exists():
            raise FileNotFoundError(f"Merge input not found: {source}")

        with source.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable line in %s", source)
                    continue

                key_val = record.get(dedup_key)
                if key_val is None:
                    raise ValueError(
                        f"Record in {source} is missing dedup key '{dedup_key}': {stripped[:200]}"
                    )
                key_str = json.dumps(key_val, sort_keys=True, ensure_ascii=False)

                if key_str in unique:
                    duplicates += 1
                    if keep == "last":
                        unique[key_str] = (stripped, i)
                else:
                    unique[key_str] = (stripped, i)

    total = len(unique)
    with target.open("w", encoding="utf-8") as out:
        for _, (line, _) in unique.items():
            out.write(line + "\n")

    logger.info(
        "Merged %d records into %s (%d duplicates removed, key=%s, keep=%s)",
        total,
        target,
        duplicates,
        dedup_key,
        keep,
    )
    return {
        "merged": total,
        "duplicates_removed": duplicates,
        "output": str(target),
    }


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
