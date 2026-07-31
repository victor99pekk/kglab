"""examples/custom_extractor.py — Using custom extraction and resolution methods.

Demonstrates how to:
1. Configure extraction mode (composed vs joint)
2. Choose entity and relation methods
3. Switch between string and embedding-based entity resolution
4. Run evaluation with an LLM for accuracy scoring

Usage:
    pip install -e ".[embeddings,llm]"
    python examples/custom_extractor.py
"""

from polygraph._shared.stage_config import ExtractionConfig, ResolutionConfig
from polygraph.pipelines import Baseline

# ── Option A: Composed extraction (spaCy entities + ontology rules) ──
print("=" * 60)
print("Option A: Composed (spaCy + ontology rules)")
print("=" * 60)

pipe_a = Baseline(
    input_paths=["data/wikipedia/"],
    output_dir="output/composed_extraction/",
    extraction=ExtractionConfig(
        mode="composed",
        entity_method="spacy",
        relation_method="ontology_rules",
        entity_options={"model_name": "en_core_web_sm"},
    ),
    resolution=ResolutionConfig(
        method="string",  # fast token-overlap matching
        threshold=0.85,
    ),
)
pipe_a.execute()

# ── Option B: Joint extraction (GraphGen LLM) ───────────────────
print("\n" + "=" * 60)
print("Option B: Joint (GraphGen LLM)")
print("=" * 60)

pipe_b = Baseline(
    input_paths=["data/wikipedia/"],
    output_dir="output/joint_extraction/",
    extraction=ExtractionConfig(
        mode="joint",
        joint_method="graphgen",
        options={"model": "gpt-4o"},
    ),
    resolution=ResolutionConfig(
        method="embedding",  # semantic similarity clustering
        threshold=0.85,
    ),
)
pipe_b.execute()

print("\nDone! Compare results in output/composed_extraction/ and output/joint_extraction/")
