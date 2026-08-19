"""Multi-variant comparison — run pipelines in a parameter matrix.

Usage::

    from kglab.benchmark_pipeline.compare import compare_variants, compare_matrix

    # Pairwise
    results = compare_variants(
        ["surface", "semantic"],
        input_paths=["data/wikipedia/"],
        output_dir="output/comparison/",
    )

    # Full matrix
    results = compare_matrix(
        variants=["surface", "semantic"],
        resolutions=["string", "embedding"],
        input_paths=["data/wikipedia/"],
        output_dir="output/matrix/",
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

from kglab._shared.stage_config import (
    ResolutionConfig,
)
from kglab.benchmark_pipeline.report import BenchmarkResult
from kglab.benchmark_pipeline.runner import BenchmarkRunner
from kglab.pipelines import PIPELINE_REGISTRY


def compare_variants(
    variants: Sequence[str],
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "output/comparison",
    **pipeline_kwargs: Any,
) -> dict[str, BenchmarkResult]:
    """Run each pipeline variant on the same input and compare results.

    Args:
        variants: Pipeline variant names (e.g. ``["surface", "semantic"]``).
        input_paths: Data files or directories.
        output_dir: Root output directory — each variant gets a subdirectory.
        **pipeline_kwargs: Forwarded to ``BenchmarkRunner`` (extraction, resolution, …).

    Returns:
        ``{variant_name: BenchmarkResult}`` dict.

    Example::

        results = compare_variants(
            ["surface", "semantic"],
            input_paths=["data/wikipedia/"],
        )
        for name, r in results.items():
            print(f"{name}: score={r.overall_score:.2f}, entities={r.num_entities}")
    """
    root = Path(output_dir)
    results: dict[str, BenchmarkResult] = {}

    for variant in variants:
        pipeline_cls = PIPELINE_REGISTRY[variant]
        variant_dir = root / variant
        runner = BenchmarkRunner(
            pipeline=pipeline_cls,
            input_paths=list(input_paths),
            output_dir=variant_dir,
            **pipeline_kwargs,
        )
        result = runner.run()
        results[variant] = result

    _write_comparison_summary(results, root)
    return results


def compare_matrix(
    variants: Sequence[str],
    input_paths: Sequence[str | Path],
    output_dir: str | Path = "output/matrix",
    *,
    resolutions: Sequence[str] | None = None,
    chunkers: Sequence[str] | None = None,
    doc_dedup_methods: Sequence[str] | None = None,
    chunk_dedup_methods: Sequence[str] | None = None,
    **pipeline_kwargs: Any,
) -> dict[str, BenchmarkResult]:
    """Run a full parameter sweep — every combination of variant × configs.

    Args:
        variants: Pipeline variant names.
        input_paths: Data files or directories.
        output_dir: Root directory — each combination gets a subdirectory.
        resolutions: Resolution methods to sweep (e.g. ``["string", "embedding"]``).
        chunkers: Chunk methods to sweep (e.g. ``["sentence", "semantic"]``).
        doc_dedup_methods: Document dedup methods to sweep (e.g. ``["minhash", "layered"]``).
        chunk_dedup_methods: Chunk dedup methods to sweep. When omitted, chunk
            dedup keeps the pipeline default (``layered``) — pass it explicitly
            if you want an embedding-free sweep.
        **pipeline_kwargs: Additional args forwarded to ``BenchmarkRunner``.

    Returns:
        ``{combo_name: BenchmarkResult}`` dict.

    Example::

        results = compare_matrix(
            variants=["surface", "semantic"],
            resolutions=["string", "embedding"],
            input_paths=["data/wikipedia/"],
        )
        # Runs 4 combinations: surface/string, surface/embedding, semantic/string, …
    """
    root = Path(output_dir)

    res_methods = list(resolutions) if resolutions else [None]  # type: ignore[list-item]
    chk_methods = list(chunkers) if chunkers else [None]  # type: ignore[list-item]
    dedup_methods = list(doc_dedup_methods) if doc_dedup_methods else [None]  # type: ignore[list-item]
    chunk_dedup_methods = (
        list(chunk_dedup_methods) if chunk_dedup_methods else [None]  # type: ignore[list-item]
    )

    results: dict[str, BenchmarkResult] = {}

    for variant, res, chk, dedup, chk_dedup in product(
        variants, res_methods, chk_methods, dedup_methods, chunk_dedup_methods
    ):
        # Build a descriptive combo name
        parts = [variant]
        if res:
            parts.append(f"res_{res}")
        if chk:
            parts.append(f"chunk_{chk}")
        if dedup:
            parts.append(f"dedup_{dedup}")
        if chk_dedup:
            parts.append(f"chunkdedup_{chk_dedup}")
        combo_name = "__".join(parts)

        pipeline_cls = PIPELINE_REGISTRY[variant]
        combo_dir = root / combo_name
        runner_kwargs: dict[str, Any] = dict(pipeline_kwargs)

        if res:
            runner_kwargs["resolution"] = ResolutionConfig(method=res)
        if chk or dedup or chk_dedup:
            from kglab._shared.stage_config import PreprocessConfig

            # Merge chunk + dedup into a single PreprocessConfig so sweeping
            # multiple axes doesn't silently overwrite one of the settings.
            preprocess_kwargs: dict[str, Any] = {}
            if chk:
                preprocess_kwargs["chunk_method"] = chk
            if dedup:
                preprocess_kwargs["doc_dedup_method"] = dedup
            if chk_dedup:
                preprocess_kwargs["chunk_dedup_method"] = chk_dedup
            runner_kwargs.setdefault("extra", {})["preprocess"] = PreprocessConfig(
                **preprocess_kwargs
            )

        runner = BenchmarkRunner(
            pipeline=pipeline_cls,
            input_paths=list(input_paths),
            output_dir=combo_dir,
            **runner_kwargs,
        )
        result = runner.run()
        results[combo_name] = result

    _write_comparison_summary(results, root)
    return results


def _write_comparison_summary(
    results: dict[str, BenchmarkResult],
    output_dir: Path,
) -> None:
    """Write a summary table comparing all runs."""
    import json

    rows: list[dict[str, Any]] = []
    for name, r in results.items():
        rows.append(
            {
                "variant": name,
                "overall_score": r.overall_score,
                "entities": r.num_entities,
                "triples": r.num_triples,
                "pipeline_variant": r.pipeline_variant,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"comparison": rows}
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
