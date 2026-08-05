"""Benchmark runner — run pipelines and compare results.

Simple usage (no YAML needed)::

    from polygraph.benchmark_pipeline import BenchmarkRunner
    from polygraph.pipelines import Baseline

    runner = BenchmarkRunner(
        pipeline=Baseline,
        input_paths=["data/wikipedia/"],
        output_dir="output/my_experiment/",
    )
    result = runner.run()
    print(f"Score: {result.overall_score:.2f}")

Compare two pipelines::

    from polygraph.benchmark_pipeline import BenchmarkRunner
    from polygraph.pipelines import Baseline

    results = BenchmarkRunner.compare(
        baseline=Baseline,
        variant=MyPipeline,
        input_paths=["data/wikipedia/"],
        output_dir="output/comparison/",
    )
    print(f"Baseline: {results['baseline'].overall_score:.2f}")
    print(f"Variant:  {results['variant'].overall_score:.2f}")

YAML config (for reproducibility)::

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
    runner = BenchmarkRunner.from_config(config)
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
    """Run a pipeline and collect standardized results.

    Two ways to use it:

    1. **Direct** — pass a pipeline class, input paths, and output dir::

        runner = BenchmarkRunner(
            pipeline=Baseline,
            input_paths=["data/"],
            output_dir="output/my_exp/",
        )
        result = runner.run()

    2. **From YAML** — for reproducible, version-controlled experiments::

        config = ExperimentConfig.from_yaml("experiments/001/config.yaml")
        runner = BenchmarkRunner.from_config(config)
        result = runner.run()

    The result is a ``BenchmarkResult`` with metrics, structural audit,
    artifact paths, and timestamps — everything you need to compare runs.
    """

    @classmethod
    def help(cls) -> str:
        """Return a guide to using BenchmarkRunner.

        Covers the two main usage modes (direct and YAML-config),
        the comparison API, and the difference between whole-pipeline
        benchmarks and stage-specific benchmarks.

        Example::

            print(BenchmarkRunner.help())
        """
        return (
            "BenchmarkRunner — Whole-Pipeline Benchmarking\n"
            "==============================================\n"
            "BenchmarkRunner runs an ENTIRE KG pipeline end-to-end and\n"
            "collects standardized results (metrics, structural audit,\n"
            "artifact paths, timestamps).\n\n"
            "This is different from stage-specific benchmarks\n"
            "(Benchmark.Dedup, Benchmark.Extraction, etc.) which test\n"
            "a single pipeline stage against a gold dataset.\n\n"
            "Two usage modes:\n\n"
            "  1. Direct (programmatic) — good for quick experiments:\n"
            "       runner = BenchmarkRunner(\n"
            "           pipeline=Baseline,\n"
            '           input_paths=["data/wikipedia/"],\n'
            '           output_dir="output/my_exp/",\n'
            "       )\n"
            "       result = runner.run()\n\n"
            "  2. YAML config — good for reproducible, version-controlled runs:\n"
            '       config = ExperimentConfig.from_yaml("experiments/001/config.yaml")\n'
            "       runner = BenchmarkRunner.from_config(config)\n"
            "       result = runner.run()\n\n"
            "Comparison mode (run two pipelines side-by-side):\n"
            "       results = BenchmarkRunner.compare(\n"
            "           baseline=Baseline,\n"
            "           variant=MyPipeline,\n"
            '           input_paths=["data/wikipedia/"],\n'
            '           output_dir="output/comparison/",\n'
            "       )\n\n"
            "Result output (per run):\n"
            "  • knowledge_graph.json  — the KG artifact\n"
            "  • metrics.json          — overall_score, completeness,\n"
            "                            consistency, duplication, entities, triples\n"
            "  • results_summary.json  — full BenchmarkResult serialized\n\n"
            "For stage-specific benchmarks (testing one component against\n"
            "gold data), see:\n"
            "  print(Benchmark.help())\n"
        )

    def __init__(
        self,
        pipeline: type[Pipeline] | None = None,
        input_paths: list[str | Path] | None = None,
        output_dir: str | Path = "output",
        *,
        ontology_path: str | Path | None = None,
        extraction: Any = None,
        resolution: Any = None,
        build: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Create a benchmark runner directly (no YAML required).

        Args:
            pipeline: A ``Pipeline`` subclass (e.g. ``Baseline``).
            input_paths: Directories or files to load.
            output_dir: Where results are written.
            ontology_path: Path to a YAML ontology file (optional).
            extraction: ``ExtractionConfig`` for custom extraction settings.
            resolution: ``ResolutionConfig`` for custom resolution settings.
            build: ``BuildConfig`` for custom graph construction settings.
            extra: Additional kwargs forwarded to the pipeline constructor.
        """
        self._pipeline_cls = pipeline
        self._input_paths = [Path(p) for p in (input_paths or [])]
        self._output_dir = Path(output_dir)
        self._ontology_path = Path(ontology_path) if ontology_path else None
        self._extraction = extraction
        self._resolution = resolution
        self._build = build
        self._extra = extra or {}
        self._config: ExperimentConfig | None = None
        self.pipeline: Pipeline | None = None

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> BenchmarkRunner:
        """Create a runner from a YAML-based experiment config."""
        runner = cls.__new__(cls)
        runner._config = config
        runner._pipeline_cls = None
        runner._input_paths = []
        runner._output_dir = Path("output")
        runner._ontology_path = None
        runner._extraction = None
        runner._resolution = None
        runner._build = None
        runner._extra = {}
        runner.pipeline = None
        return runner

    # ── Comparison ──────────────────────────────────────────────

    @staticmethod
    def compare(
        baseline: type[Pipeline],
        variant: type[Pipeline],
        input_paths: list[str | Path],
        output_dir: str | Path = "output/comparison",
        *,
        baseline_kwargs: dict[str, Any] | None = None,
        variant_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, BenchmarkResult]:
        """Run two pipelines side by side and return both results.

        Args:
            baseline: The reference pipeline class (e.g. ``Baseline``).
            variant: The new pipeline class to compare against baseline.
            input_paths: Shared input data for both pipelines.
            output_dir: Parent directory — each pipeline gets a subdirectory.
            baseline_kwargs: Extra kwargs for the baseline pipeline constructor.
            variant_kwargs: Extra kwargs for the variant pipeline constructor.

        Returns:
            Dict with keys ``"baseline"`` and ``"variant"``, each mapping to
            the respective ``BenchmarkResult``.

        Example::

            results = BenchmarkRunner.compare(
                baseline=Baseline,
                variant=LLMPipeline,
                input_paths=["data/wikipedia/"],
                output_dir="output/llm_vs_baseline/",
            )
            improvement = results["variant"].overall_score - results["baseline"].overall_score
            print(f"LLM pipeline improved score by {improvement:+.2f}")
        """
        base_dir = Path(output_dir)
        results: dict[str, BenchmarkResult] = {}

        for label, cls, kwargs in [
            ("baseline", baseline, baseline_kwargs or {}),
            ("variant", variant, variant_kwargs or {}),
        ]:
            runner = BenchmarkRunner(
                pipeline=cls,
                input_paths=[str(p) for p in input_paths],
                output_dir=base_dir / label,
                **kwargs,
            )
            results[label] = runner.run()

        return results

    # ── Public API ──────────────────────────────────────────────

    def run(self) -> BenchmarkResult:
        """Run the pipeline and return aggregated results.

        If constructed from a YAML config, uses that config's settings.
        Otherwise uses the direct parameters passed to ``__init__``.
        """
        if self._config is not None:
            return self._run_from_config()

        return self._run_direct()

    # ── Direct execution ────────────────────────────────────────

    def _run_direct(self) -> BenchmarkResult:
        if self._pipeline_cls is None:
            raise ValueError(
                "No pipeline specified. Pass pipeline= to BenchmarkRunner() "
                "or use BenchmarkRunner.from_config()."
            )
        if not self._input_paths:
            raise ValueError("No input paths specified. Pass input_paths= to BenchmarkRunner().")

        output_dir = self._output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_kwargs: dict[str, Any] = {}
        if self._ontology_path:
            pipeline_kwargs["ontology_path"] = str(self._ontology_path)
        pipeline_kwargs.update(self._extra)

        pipeline = self._pipeline_cls(
            extraction=self._extraction,
            resolution=self._resolution,
            build=self._build,
            **pipeline_kwargs,
        )
        self.pipeline = pipeline

        t0 = time.perf_counter()
        pipeline.execute(
            input_paths=[str(p) for p in self._input_paths],
            output_dir=str(output_dir),
        )
        elapsed_s = time.perf_counter() - t0

        return self._collect_results(
            output_dir=output_dir,
            elapsed_s=elapsed_s,
            pipeline_variant=self._pipeline_cls.__name__,
            neo4j_uploaded=False,
            neo4j_cleared=False,
        )

    # ── Config-driven execution ─────────────────────────────────

    def _run_from_config(self) -> BenchmarkResult:
        config = self._config
        assert config is not None

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
        # A YAML ``preprocess:`` block arrives as a raw dict via ``extra``;
        # coerce it to the typed config the pipeline expects.
        raw_preprocess = pipeline_kwargs.get("preprocess")
        if isinstance(raw_preprocess, dict):
            from polygraph._shared.stage_config import PreprocessConfig

            pipeline_kwargs["preprocess"] = PreprocessConfig.from_dict(raw_preprocess)

        pipeline: Pipeline = pipeline_cls(
            extraction=config.extraction,
            resolution=config.resolution,
            build=config.build,
            **pipeline_kwargs,
        )

        # 3. Execute
        self.pipeline = pipeline
        t0 = time.perf_counter()
        pipeline.execute(
            input_paths=[str(p) for p in config.input_paths],
            output_dir=str(output_dir),
        )
        elapsed_s = time.perf_counter() - t0

        # 4. Optional Neo4j upload
        neo4j_uploaded = False
        if config.upload_neo4j:
            pipeline.upload_to_neo4j(clear=config.clear_neo4j)
            neo4j_uploaded = True

        return self._collect_results(
            output_dir=output_dir,
            elapsed_s=elapsed_s,
            pipeline_variant=config.pipeline_variant,
            neo4j_uploaded=neo4j_uploaded,
            neo4j_cleared=config.clear_neo4j,
        )

    # ── Shared result collection ────────────────────────────────

    def _collect_results(
        self,
        output_dir: Path,
        elapsed_s: float,
        pipeline_variant: str,
        neo4j_uploaded: bool,
        neo4j_cleared: bool,
    ) -> BenchmarkResult:
        """Read back metrics and assemble the BenchmarkResult."""
        metrics_path = output_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        structural_audit: dict[str, Any] = {}
        if metrics_path.exists():
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            structural_audit = loaded.pop("structural_audit", {})
            metrics = loaded

        # Write full entity duplicate pairs to a separate file
        all_dupes = structural_audit.get("entity_duplication", {}).pop("all_duplicate_pairs", None)
        if all_dupes:
            dupes_path = output_dir / "entity_duplicates.json"
            dupes_path.write_text(json.dumps(all_dupes, indent=2, default=str))
            print(f"[benchmark] entity duplicates → {dupes_path}")

        metrics["wall_time_s"] = round(elapsed_s, 2)

        # Discover generated artifacts
        artifacts: dict[str, str] = {}
        for candidate in output_dir.iterdir():
            if candidate.is_file() and candidate.suffix in (".json", ".graphml"):
                if candidate.name == "metrics.json":
                    continue
                artifacts[candidate.suffix.lstrip(".")] = str(candidate)

        config_snapshot: dict[str, Any] = {}
        if self._config is not None:
            config_snapshot = {
                "description": self._config.description,
                "ontology_path": str(self._config.ontology_path)
                if self._config.ontology_path
                else None,
                "llm_judge": self._config.llm_judge,
            }

        result = BenchmarkResult(
            experiment_name=self._config.name if self._config else pipeline_variant,
            pipeline_variant=pipeline_variant,
            input_paths=[str(p) for p in (self._input_paths or [])],
            output_dir=str(output_dir),
            metrics=metrics,
            structural_audit=structural_audit,
            artifacts=artifacts,
            neo4j_uploaded=neo4j_uploaded,
            neo4j_cleared=neo4j_cleared,
            config_snapshot=config_snapshot,
        )

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
