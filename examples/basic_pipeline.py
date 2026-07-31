"""examples/basic_pipeline.py — Simplest possible KG generation with Polygraph.

This script:
1. Loads input documents from a directory
2. Runs the Baseline pipeline (preprocess → extract → resolve → build → eval → export)
3. Writes the KG JSON and metrics to an output directory

Usage:
    python examples/basic_pipeline.py
"""

from polygraph.pipelines import Baseline

# ── Configuration ───────────────────────────────────────────────
INPUT_DIR = "data/wikipedia/"  # directory with .jsonl files
OUTPUT_DIR = "output/basic_pipeline/"

# ── Run ─────────────────────────────────────────────────────────
pipe = Baseline(
    input_paths=[INPUT_DIR],
    output_dir=OUTPUT_DIR,
)
pipe.execute()

print(f"\nDone! Results in {OUTPUT_DIR}")
print("  knowledge_graph.json — full KG with nodes, edges, entities, triples")
print("  metrics.json         — quality evaluation scores")
