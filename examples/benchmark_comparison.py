"""examples/benchmark_comparison.py — Compare two pipelines side by side.

Demonstrates BenchmarkRunner.compare() for direct side-by-side comparison
without needing YAML config files.

Usage:
    python examples/benchmark_comparison.py
"""

from polygraph.benchmark_pipeline import BenchmarkRunner
from polygraph.pipelines import Baseline

# ── Option A: Direct comparison (simplest) ──────────────────────
print("=" * 60)
print("Comparing pipelines directly")
print("=" * 60)

results = BenchmarkRunner.compare(
    baseline=Baseline,
    variant=Baseline,  # replace with your custom pipeline class
    input_paths=["data/wikipedia/"],
    output_dir="output/comparison/",
)

for label, result in results.items():
    print(f"\n{label}:")
    print(f"  Overall score: {result.overall_score:.2f}")
    print(f"  Num entities:  {result.num_entities}")
    print(f"  Num triples:   {result.num_triples}")

# ── Option B: Single pipeline run ───────────────────────────────
print("\n" + "=" * 60)
print("Running a single pipeline")
print("=" * 60)

runner = BenchmarkRunner(
    pipeline=Baseline,
    input_paths=["data/wikipedia/"],
    output_dir="output/single_run/",
)
result = runner.run()

print(f"\nScore: {result.overall_score:.2f}")
print(f"Entities: {result.num_entities}")
print(f"Triples: {result.num_triples}")
