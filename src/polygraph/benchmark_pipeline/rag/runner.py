"""RAG benchmark — test KG quality for retrieval-augmented generation.

Each pipeline builds a KG, then retrieval accuracy is measured against
gold query-answer pairs.  Tests whether the KG can surface the right
entities, relations, and source chunks for downstream RAG.

Gold dataset format (JSONL)::

    {"query": "Who discovered radium?", "answer_entity": "Marie Curie", "supporting_chunks": ["chunk:0", "chunk:3"]}
    {"query": "Where was Einstein born?", "answer_entity": "Ulm", "relation_path": ["born_in"]}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from polygraph._shared.types import PreprocessResult
from polygraph.data import Data
from polygraph.pipelines import Pipeline

#: Default location for the gold RAG dataset when no ``dataset`` is passed.
_DEFAULT_DATASET = "benchmarks/data/rag_gold.jsonl"

#: Default retrieval depth used for recall@k / precision@k.
_DEFAULT_K = 10

#: Cap on gold queries scored per run — keeps large gold sets (e.g. the
#: ~90k-record HotpotQA train set) tractable.
_DEFAULT_MAX_QUERIES = 200


class RAGRunner:
    """Benchmark KG retrieval quality for RAG use cases.

    Each pipeline is executed on the gold documents, then retrieval
    metrics are computed against the gold query-answer pairs.

    Tests whether the KG can surface the right entities, relations,
    and source chunks for downstream retrieval-augmented generation.
    """

    def __init__(self, dataset: str | Path = _DEFAULT_DATASET) -> None:
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
        *,
        k: int = _DEFAULT_K,
        max_queries: int = _DEFAULT_MAX_QUERIES,
    ) -> dict[str, Any]:
        """Run RAG benchmark and return retrieval metrics per pipeline.

        Each pipeline is executed on the gold documents, then retrieval
        metrics are computed against the gold query-answer pairs.

        Args:
            pipelines: ``{name: Pipeline}`` dict.
            input_paths: Documents to build the KG from.  If not provided,
                the input must already be set on each pipeline instance.
            k: Retrieval depth for entity recall@k / chunk precision@k.
            max_queries: Maximum number of gold queries to score.  The gold
                file may be huge (HotpotQA train is ~90k records); this caps
                the run while keeping the metric honest.

        Returns:
            ``{pipeline_name: {entity_recall_at_k, entity_coverage,
            chunk_recall_at_k, chunk_precision_at_k, relation_path_accuracy,
            precision, recall, f1, runtime_seconds, n_queries}}`` — one
            entry per pipeline.
        """
        Data.download("bench_rag", path=str(self.dataset))
        gold = _load_gold(self.dataset)
        if max_queries and len(gold) > max_queries:
            gold = gold[:max_queries]

        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            try:
                if input_paths:
                    pipeline.input_paths = [Path(p) for p in input_paths]
                if not pipeline.input_paths:
                    raise ValueError(f"RAG benchmark needs input_paths for pipeline '{name}'")

                t0 = time.perf_counter()
                result = pipeline.preprocess()
                kg = pipeline.build_kg(result)
                elapsed_s = time.perf_counter() - t0

                chunk_docs = result.chunks if isinstance(result, PreprocessResult) else result
                metrics = _evaluate(kg, chunk_docs, gold, k=k)
                metrics["runtime_seconds"] = round(elapsed_s, 4)
                metrics["n_queries"] = len(gold)
                # run_benchmarks.py _format() requires precision/recall/f1 keys.
                metrics["precision"] = metrics["chunk_precision_at_k"]
                metrics["recall"] = metrics["chunk_recall_at_k"]
                p, r = metrics["precision"], metrics["recall"]
                metrics["f1"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
                results[name] = metrics
            except Exception as exc:
                results[name] = {
                    "entity_recall_at_k": 0.0,
                    "entity_coverage": 0.0,
                    "chunk_recall_at_k": 0.0,
                    "chunk_precision_at_k": 0.0,
                    "relation_path_accuracy": None,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "runtime_seconds": 0.0,
                    "n_queries": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return results


# ── Helpers ─────────────────────────────────────────────────────


def _load_gold(path: Path) -> list[dict[str, Any]]:
    """Load gold query-answer records from a JSONL file.

    Args:
        path: Path to the gold JSONL file.

    Returns:
        List of ``{"query", "answer_entity", "supporting_chunks",
        "relation_path"}`` records.

    Raises:
        ValueError: If a record is malformed or the file has no records.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "query" not in record:
                raise ValueError(f"{path}:{line_no} — record must have a 'query' key")
            records.append(
                {
                    "query": str(record.get("query", "")),
                    "answer_entity": str(record.get("answer_entity", "")).strip(),
                    "supporting_chunks": list(record.get("supporting_chunks", [])),
                    "relation_path": list(record.get("relation_path", [])),
                }
            )
    if not records:
        raise ValueError(f"No gold records found in {path}")
    return records


def _normalize(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _token_overlap(query: str, text: str) -> int:
    """Count query tokens that also appear in *text* (case-insensitive)."""
    query_tokens = set(query.casefold().split())
    text_tokens = set(text.casefold().split())
    if not query_tokens:
        return 0
    return len(query_tokens & text_tokens)


def _relation_name(triple: Any) -> str:
    """Extract a normalized relation/predicate name from a triple."""
    if isinstance(triple, (tuple, list)) and len(triple) >= 3:
        predicate = triple[1]
    elif isinstance(triple, dict):
        predicate = triple.get("predicate", triple.get("relation", triple.get("type", "")))
    else:
        predicate = triple
    if isinstance(predicate, dict):
        predicate = predicate.get("name", predicate.get("type", predicate.get("label", "")))
    return _normalize(predicate).casefold()


def _evaluate(
    kg: dict[str, Any],
    chunk_docs: list[Any],
    gold: list[dict[str, Any]],
    *,
    k: int,
) -> dict[str, Any]:
    """Score KG retrieval quality against gold query-answer pairs.

    * ``entity_recall_at_k`` — fraction of queries whose answer entity
      appears among the top-*k* entities ranked by token overlap with the query.
    * ``entity_coverage`` — fraction of queries whose answer entity exists
      anywhere in the KG (recall@all).
    * ``chunk_recall_at_k`` — fraction of gold supporting chunks found among
      the top-*k* chunks ranked by token overlap with the query.
    * ``chunk_precision_at_k`` — fraction of the top-*k* chunks that are gold
      supporting chunks.
    * ``relation_path_accuracy`` — fraction of gold relation paths that are
      fully present in the KG (``None`` when gold has no relation paths).
    """
    entities = kg.get("entities", [])
    triples = kg.get("triples", [])
    entity_names = [str(e.get("name", "")) for e in entities]
    chunk_texts = [_normalize(c.content) for c in chunk_docs if c.content.strip()]
    kg_relations = {_relation_name(t) for t in triples}

    entity_recall_at_k = 0.0
    entity_coverage = 0.0
    chunk_recall_at_k = 0.0
    chunk_precision_at_k = 0.0
    relation_total = 0
    relation_hit = 0

    for record in gold:
        query = record["query"]
        answer = record["answer_entity"].casefold().strip()
        gold_chunks = [_normalize(c) for c in record["supporting_chunks"] if str(c).strip()]

        # Entity retrieval — rank by token overlap with the query.
        ranked_entities = sorted(
            entity_names, key=lambda name: _token_overlap(query, name), reverse=True
        )
        top_entities = ranked_entities[:k]
        entity_recall_at_k += float(
            bool(answer) and any(answer in name.casefold() for name in top_entities)
        )
        entity_coverage += float(
            bool(answer) and any(answer in name.casefold() for name in entity_names)
        )

        # Chunk retrieval — rank chunks by token overlap with the query.
        ranked_chunks = sorted(
            chunk_texts, key=lambda text: _token_overlap(query, text), reverse=True
        )
        top_chunks = ranked_chunks[:k]
        if gold_chunks:
            found = sum(1 for g in gold_chunks if any(g and g in t for t in top_chunks))
            chunk_recall_at_k += found / len(gold_chunks)
        if top_chunks:
            supporting = sum(1 for t in top_chunks if any(g and g in t for g in gold_chunks))
            chunk_precision_at_k += supporting / len(top_chunks)

        # Relation path accuracy — every relation in the gold path present in KG.
        relation_path = record["relation_path"]
        if relation_path:
            relation_total += 1
            if all(_normalize(r).casefold() in kg_relations for r in relation_path):
                relation_hit += 1

    n = len(gold)
    return {
        "entity_recall_at_k": round(entity_recall_at_k / n, 4) if n else 0.0,
        "entity_coverage": round(entity_coverage / n, 4) if n else 0.0,
        "chunk_recall_at_k": round(chunk_recall_at_k / n, 4) if n else 0.0,
        "chunk_precision_at_k": round(chunk_precision_at_k / n, 4) if n else 0.0,
        "relation_path_accuracy": round(relation_hit / relation_total, 4)
        if relation_total
        else None,
    }
