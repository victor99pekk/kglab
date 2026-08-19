"""Semantic pipeline — embedding-based resolution for richer entity merging.

The upgrade from ``Baseline``: larger spaCy model, embedding resolution
instead of string matching, and composite document relations.  Zero external
APIs — everything runs locally.

Usage::

    from kglab.pipelines.local import Semantic

    pipe = Semantic(
        input_paths=["data/wikipedia/connected.jsonl"],
        output_dir="generated_KGs/",
    )
    pipe.execute()

    # Or via CLI:
    # python main.py --variant semantic -i data/wikipedia/connected.jsonl
"""

from __future__ import annotations

from typing import Any

from kglab._shared.stage_config import (
    DocumentRelationConfig,
    ExtractionConfig,
    ResolutionConfig,
)
from kglab.pipelines.baseline import Baseline


class Semantic(Baseline):
    """Pre-configured pipeline with embedding resolution and richer extraction.

    Compared to ``Baseline``:

    * Entity extraction: ``en_core_web_lg`` (larger spaCy model with word vectors)
      instead of ``en_core_web_sm``.
    * Entity resolution: ``embedding`` (semantic similarity via sentence-transformers)
      instead of ``string`` (token overlap).
    * Document relations: ``hyperlink`` + ``shared_references`` instead of just
      ``hyperlink``.
    * Preprocessing: same as Baseline (semantic chunking, layered dedup, ftfy cleaning).

    All components run locally — no LLM API required.  The only additional
    dependency is ``en_core_web_lg`` (install with ``python -m spacy download
    en_core_web_lg``).

    Args:
        input_paths: Data files or directories to load.
        output_dir: Where results are written.
        **kwargs: Forwarded to ``Baseline`` (supports all the same keyword args
            for further customization).
    """

    def __init__(self, **kwargs: Any) -> None:
        # ── Only override configs the user hasn't explicitly passed ──
        if "extraction" not in kwargs:
            kwargs["extraction"] = ExtractionConfig(
                mode="composed",
                entity_method="spacy",
                entity_options={"model_name": "en_core_web_lg"},
                relation_method="ontology_rules",
                document_relation=DocumentRelationConfig(
                    methods=["hyperlink", "shared_references"],
                ),
            )

        if "resolution" not in kwargs:
            kwargs["resolution"] = ResolutionConfig(
                method="embedding",
                threshold=0.80,
            )

        super().__init__(**kwargs)
