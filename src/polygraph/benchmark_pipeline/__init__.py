"""Benchmark pipeline — standardized experiment runner for KG quality evaluation.

Wraps any ``Pipeline`` subclass and produces reproducible, self-documenting
experiment outputs: metrics, graph files, and a ``results_summary.json``.

Usage:
    from polygraph.benchmark_pipeline import BenchmarkRunner
    from polygraph.benchmark_pipeline.config import ExperimentConfig

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
    runner = BenchmarkRunner(config)
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
