"""kg-generator — Research toolkit for building high-quality KGs for LLM training.

Modules:
    data/         — Download & enrich supported datasets (polygraph.data.Data)
    preprocess/   — Raw text → clean, deduplicated chunks
    kg_build/     — Chunks → Knowledge Graph (extract → resolve → build)
    kg_eval/      — KG quality evaluation & structural audits
    kg_export/    — KG → files / Neo4j
    kg_update/    — Add new documents to existing KGs
    finetune/     — KG → training data → fine-tuned LLM
    _shared/      — Config, identity, and common types (zero dependencies)
"""

__version__ = "0.2.0"
