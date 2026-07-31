"""Benchmark pipeline — run and compare KG generation pipelines.

Wraps any ``Pipeline`` subclass and produces reproducible, self-documenting
experiment outputs: metrics, graph files, and a ``results_summary.json``.

Direct API (no YAML needed)::

    from polygraph.benchmark_pipeline import BenchmarkRunner
    from polygraph.pipelines import Baseline

    runner = BenchmarkRunner(
        pipeline=Baseline,
        input_paths=["data/wikipedia/"],
        output_dir="output/my_exp/",
    )
    result = runner.run()
    print(result.overall_score, result.num_entities, result.num_triples)

Side-by-side comparison::

    results = BenchmarkRunner.compare(
        baseline=Baseline,
        variant=MyPipeline,
        input_paths=["data/wikipedia/"],
        output_dir="output/comparison/",
    )

YAML config (for reproducibility)::

    from polygraph.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
    runner = BenchmarkRunner.from_config(config)
    result = runner.run()
"""

from polygraph.benchmark_pipeline.config import ExperimentConfig
from polygraph.benchmark_pipeline.report import BenchmarkResult, write_report
from polygraph.benchmark_pipeline.runner import BenchmarkRunner

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "ExperimentConfig",
    "write_report",
]
