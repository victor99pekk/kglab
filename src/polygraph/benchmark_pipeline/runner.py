"""Benchmark runner — orchestrates a pipeline run and collects results.

Usage:
    from polygraph.benchmark_pipeline import BenchmarkRunner
    from polygraph.benchmark_pipeline.config import ExperimentConfig

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
    runner = BenchmarkRunner(config)
    result = runner.run()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from polygraph.benchmark_pipeline.config import ExperimentConfig
from polygraph.benchmark_pipeline.report import BenchmarkResult, write_report
from polygraph.pipelines import PIPELINE_REGISTRY, Pipeline


class BenchmarkRunner:
    """Takes an experiment config, runs the specified pipeline, and writes a
    standardized results summary.

    The runner:

    1. Resolves the pipeline class from ``pipeline.variant`` in the config.
    2. Instantiates the pipeline with input paths, output dir, and ontology.
    3. Calls ``pipeline.execute()`` (preprocess → build → evaluate → export).
    4. Optionally uploads to Neo4j.
    5. Reads back the generated ``metrics.json`` and assembles a
       ``BenchmarkResult`` written as ``results_summary.json``.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    # ── Public API ──────────────────────────────────────────────

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark pipeline and return aggregated results."""
        config = self._config
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 0. Ensure dataset is downloaded (cached if already present)
        if config.dataset_name:
            from polygraph.data import Data

            if not config.input_paths:
                raise ValueError(
                    f"Experiment specifies dataset '{config.dataset_name}' "
                    "but has no input.paths to download to."
                )
            dataset_path = config.input_paths[0]
            dataset_path = Path(dataset_path)
            if dataset_path.is_dir():
                dataset_path = dataset_path / "articles.jsonl"

            params = dict(config.dataset_params)
            Data.download(
                config.dataset_name,
                path=str(dataset_path),
                enrich=True,
                **params,
            )

        # 1. Resolve pipeline class
        pipeline_cls = self._resolve_pipeline(config.pipeline_variant)

        # 2. Instantiate pipeline with typed configs
        pipeline_kwargs: dict[str, Any] = {}
        if config.ontology_path:
            pipeline_kwargs["ontology_path"] = str(config.ontology_path)
        pipeline_kwargs.update(config.extra)

        pipeline: Pipeline = pipeline_cls(
            input_paths=[str(p) for p in config.input_paths],
            output_dir=str(output_dir),
            extraction=config.extraction,
            resolution=config.resolution,
            build=config.build,
            **pipeline_kwargs,
        )

        # 3. Execute
        self.pipeline = pipeline
        t0 = time.perf_counter()
        pipeline.execute()
        elapsed_s = time.perf_counter() - t0

        # 4. Optional Neo4j upload
        neo4j_uploaded = False
        if config.upload_neo4j:
            pipeline.upload_to_neo4j(clear=config.clear_neo4j)
            neo4j_uploaded = True

        # 5. Read back metrics
        metrics_path = output_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        structural_audit: dict[str, Any] = {}
        if metrics_path.exists():
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            structural_audit = loaded.pop("structural_audit", {})
            metrics = loaded

        # Write full entity duplicate pairs to a separate file (summary caps at 20)
        all_dupes = structural_audit.get("entity_duplication", {}).pop("all_duplicate_pairs", None)
        if all_dupes:
            dupes_path = output_dir / "entity_duplicates.json"
            dupes_path.write_text(json.dumps(all_dupes, indent=2, default=str))
            print(f"[benchmark] entity duplicates → {dupes_path}")

        metrics["wall_time_s"] = round(elapsed_s, 2)

        # 6. Discover generated artifacts
        artifacts: dict[str, str] = {}
        for candidate in output_dir.iterdir():
            if candidate.is_file() and candidate.suffix in (".json", ".graphml"):
                if candidate.name == "metrics.json":
                    continue
                artifacts[candidate.suffix.lstrip(".")] = str(candidate)

        # 7. Build result
        result = BenchmarkResult(
            experiment_name=config.name,
            pipeline_variant=config.pipeline_variant,
            input_paths=[str(p) for p in config.input_paths],
            output_dir=str(output_dir),
            metrics=metrics,
            structural_audit=structural_audit,
            artifacts=artifacts,
            neo4j_uploaded=neo4j_uploaded,
            neo4j_cleared=config.clear_neo4j,
            config_snapshot={
                "description": config.description,
                "ontology_path": str(config.ontology_path) if config.ontology_path else None,
                "llm_judge": config.llm_judge,
            },
        )

        # 8. Write report
        report_path = write_report(result, output_dir)
        print(f"[benchmark] results_summary → {report_path}")
        return result

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _resolve_pipeline(variant: str) -> type[Pipeline]:
        """Look up a pipeline class by its registered variant name."""
        cls = PIPELINE_REGISTRY.get(variant)
        if cls is None:
            available = ", ".join(sorted(PIPELINE_REGISTRY.keys()))
            raise ValueError(f"Unknown pipeline variant: '{variant}'. Available: {available}")
        return cls
