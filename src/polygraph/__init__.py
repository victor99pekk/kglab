"""kg-generator — Research toolkit for building high-quality KGs for LLM training.

Quick start::

    from polygraph.pipelines import Baseline

    pipe = Baseline(input_paths=["data/"], output_dir="output/")
    pipe.execute()  # preprocess → build_kg → evaluate → export

Key modules::

    pipelines/    — Swappable KG pipeline variants (Baseline + custom)
    preprocess/   — Raw text → clean, deduplicated chunks (load, clean, chunk, quality, dedup)
    kg_build/     — Chunks → Knowledge Graph (extract → resolve → build)
    kg_eval/      — KG quality evaluation & structural audits
    kg_export/    — KG → JSON / GraphML / Neo4j / RDF / Cytoscape
    benchmark_pipeline/ — Reproducible experiment runner with config-driven results
    finetune/     — KG → QA training data → fine-tuned LLM
    data/         — Download & enrich supported datasets
    _shared/      — Config, identity, and common types (zero dependencies)

See ``docs/tutorial.md`` for a full walkthrough and ``docs/api_reference.md``
for the complete API.
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
