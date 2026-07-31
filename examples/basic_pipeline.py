"""examples/basic_pipeline.py — Simplest possible KG generation with Polygraph.

Demonstrates the three tiers of preprocessing customization:

  Tier 0 — Zero config (sensible defaults)
  Tier 1 — Simple knobs (chunk_method, dedup thresholds, etc.)
  Tier 2 — Full stage control (reorder, insert, skip stages)
  Tier 3 — Custom Preprocessor subclass (not shown here, see tutorial)

Usage:
    python examples/basic_pipeline.py
"""

from polygraph._shared.stage_config import PreprocessConfig, PreprocessStage
from polygraph.pipelines import Baseline

INPUT_DIR = "data/wikipedia/"  # directory with .jsonl files

# ── Tier 0: Zero config ────────────────────────────────────────
print("=" * 60)
print("Tier 0: Zero config (sensible defaults)")
print("=" * 60)

pipe = Baseline(
    input_paths=[INPUT_DIR],
    output_dir="output/basic_pipeline/",
)
pipe.execute()

print("\nDone! Results in output/basic_pipeline/")
print("  knowledge_graph.json — full KG with nodes, edges, entities, triples")
print("  metrics.json         — quality evaluation scores")

# ── Tier 1: Simple knobs ───────────────────────────────────────
print("\n" + "=" * 60)
print("Tier 1: Simple knobs (semantic chunking, higher dedup)")
print("=" * 60)

pipe = Baseline(
    input_paths=[INPUT_DIR],
    output_dir="output/semantic_chunks/",
    preprocess=PreprocessConfig(
        chunk_method="semantic",
        chunk_target_tokens=300,
        chunk_dedup_threshold=0.90,
    ),
)
pipe.execute()

# ── Tier 2: Full stage control ─────────────────────────────────
print("\n" + "=" * 60)
print("Tier 2: Full stage control (skip quality, reorder)")
print("=" * 60)

pipe = Baseline(
    input_paths=[INPUT_DIR],
    output_dir="output/custom_stages/",
    preprocess=PreprocessConfig(
        stages=[
            PreprocessStage("load", "baseline"),
            PreprocessStage("chunk", "fixed", options={"size": 500, "overlap": 100}),
            PreprocessStage("dedup", "minhash", options={"threshold": 0.90}),
            # quality and link_normalize stages skipped
        ]
    ),
)
pipe.execute()

print("\nAll pipelines done!")
print("Compare output/basic_pipeline/ vs output/semantic_chunks/ vs output/custom_stages/")
