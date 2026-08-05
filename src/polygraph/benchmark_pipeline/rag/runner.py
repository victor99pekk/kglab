"""RAG benchmark — test KG quality for retrieval-augmented generation.

Each pipeline builds a KG, then retrieval accuracy is measured against
gold query-answer pairs.  Tests whether the KG can surface the right
entities, relations, and source chunks for downstream RAG.

Gold dataset format (JSONL)::

    {"query": "Who discovered radium?", "answer_entity": "Marie Curie", "supporting_chunks": ["chunk:0", "chunk:3"]}
    {"query": "Where was Einstein born?", "answer_entity": "Ulm", "relation_path": ["born_in"]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polygraph.pipelines import Pipeline


class RAGRunner:
    """Benchmark KG retrieval quality for RAG use cases.

    Each pipeline is executed on the gold documents, then retrieval
    metrics are computed against the gold query-answer pairs.

    Tests whether the KG can surface the right entities, relations,
    and source chunks for downstream retrieval-augmented generation.
    """

    def __init__(self, dataset: str | Path) -> None:
        self.dataset = Path(dataset)

    @classmethod
    def help(cls) -> str:
        """Return a human-readable description of this benchmark.

        Use this to understand what the benchmark measures, the gold
        dataset format, and the metrics it produces — without reading
        the source code.

        Example::

            print(RAGRunner.help())
        """
        return (
            "RAG (Retrieval-Augmented Generation) Benchmark\n"
            "==============================================\n"
            "What it tests:\n"
            "  Whether a KG built by a pipeline can effectively support\n"
            "  retrieval-augmented generation. Given a query, can the KG\n"
            "  surface the correct entities, relation paths, and source\n"
            "  chunks that a downstream LLM needs to answer accurately?\n\n"
            "Why it matters:\n"
            "  KGs are often built to power RAG applications. This\n"
            "  benchmark measures end-to-end retrieval quality — the\n"
            "  ultimate test of whether your KG is useful for QA.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"query": "Who discovered radium?", "answer_entity": "Marie Curie",\n'
            '   "supporting_chunks": ["chunk:0", "chunk:3"]}\n'
            '  {"query": "Where was Einstein born?", "answer_entity": "Ulm",\n'
            '   "relation_path": ["born_in"]}\n\n'
            "Primary metrics:\n"
            "  Entity recall@k — does the correct answer entity appear in top-k results?\n"
            "  Relation path accuracy — does the KG contain the correct relation chain?\n"
            "  Chunk retrieval precision — are supporting chunks correctly retrieved?\n"
        )

    def run(
        self,
        pipelines: dict[str, Pipeline],
        input_paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        """Run RAG benchmark and return retrieval metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.
            input_paths: Documents to build the KG from.  If not provided,
                the input must already be set on each pipeline instance.
        """
        # TODO: implement
        # 1. Build KG from each pipeline (run pipeline if needed)
        # 2. Load gold query-answer pairs from self.dataset
        # 3. For each query, retrieve from the KG:
        #    - Entity recall@k
        #    - Relation path accuracy
        #    - Supporting chunk retrieval
        # 4. Compute retrieval metrics per pipeline
        raise NotImplementedError("RAG benchmark not yet implemented")
