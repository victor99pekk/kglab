"""Benchmark pipeline — run and compare KG generation pipelines.

Targeted stage benchmarks (each tests one stage with gold data)::

    from polygraph.benchmark_pipeline import Benchmark
    from polygraph.pipelines import Baseline, Semantic
    from polygraph._shared.stage_config import PreprocessConfig, ResolutionConfig

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

    from polygraph.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
    runner = BenchmarkRunner.from_config(config)
    result = runner.run()
"""

from polygraph.benchmark_pipeline.benchmark import Benchmark
from polygraph.benchmark_pipeline.config import ExperimentConfig
from polygraph.benchmark_pipeline.report import BenchmarkResult, write_report
from polygraph.benchmark_pipeline.runner import BenchmarkRunner

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ExperimentConfig",
    "write_report",
]
