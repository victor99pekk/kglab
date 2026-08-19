"""Benchmark pipeline — run and compare KG generation pipelines.

Targeted stage benchmarks (each tests one stage with gold data)::

    from kglab.benchmark_pipeline import Benchmark
    from kglab.pipelines import Baseline, Semantic
    from kglab._shared.stage_config import PreprocessConfig, ResolutionConfig

    result = Benchmark.Dedup(dataset="benchmarks/data/dedup_gold.jsonl").run(
        pipelines={
            "surface": Baseline(),
            "surface_layered": Baseline(preprocess=PreprocessConfig(doc_dedup_method="layered")),
            "semantic": Semantic(),
        }
    )

    result = Benchmark.Resolution(dataset="benchmarks/data/resolution_gold.jsonl").run(
        pipelines={
            "string": Baseline(),
            "embed": Baseline(resolution=ResolutionConfig(method="embedding")),
        }
    )

Single pipeline (with YAML config for reproducibility)::

    from kglab.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
    runner = BenchmarkRunner.from_config(config)
    result = runner.run()
"""

from kglab.benchmark_pipeline.benchmark import Benchmark
from kglab.benchmark_pipeline.config import ExperimentConfig
from kglab.benchmark_pipeline.render import format_stage, format_stages
from kglab.benchmark_pipeline.report import BenchmarkResult, StageResult, write_report
from kglab.benchmark_pipeline.runner import BenchmarkRunner

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ExperimentConfig",
    "StageResult",
    "format_stage",
    "format_stages",
    "write_report",
]
