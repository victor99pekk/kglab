"""examples/benchmark_comparison.py — Reproducible experiment comparison.

Demonstrates how to use BenchmarkRunner to run multiple pipeline variants
from YAML config files and compare results.

Usage:
    python examples/benchmark_comparison.py
"""

from pathlib import Path

from polygraph.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

# ── Run two experiments from config files ────────────────────────
experiments_dir = Path("experiments/kg/")

for exp_dir in sorted(experiments_dir.glob("*")):
    if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
        continue

    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"  [skip] {exp_dir.name} — no config.yaml")
        continue

    print(f"\n{'=' * 60}")
    print(f"Running: {exp_dir.name}")
    print(f"{'=' * 60}")

    config = ExperimentConfig.from_yaml(config_path)
    runner = BenchmarkRunner(config)
    result = runner.run()

    print(f"  Overall score:  {result.overall_score:.2f}")
    print(f"  Num entities:   {result.num_entities}")
    print(f"  Num triples:    {result.num_triples}")
    print(f"  Results → {config.output_dir / 'results_summary.json'}")

print("\nAll experiments complete!")
