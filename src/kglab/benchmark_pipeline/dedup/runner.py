"""Deduplication benchmark — compare pipeline dedup quality against gold pairs.

Each pipeline is run on the gold dataset and its dedup decisions are scored
against known duplicate / non-duplicate labels.

Gold dataset format (JSONL)::

    {"text_a": "...", "text_b": "...", "label": "duplicate"}
    {"text_a": "...", "text_b": "...", "label": "not_duplicate"}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kglab._shared import Document
from kglab.benchmark_pipeline.report import StageResult
from kglab.data import Data
from kglab.pipelines import Pipeline
from kglab.preprocess.dedup import Deduplicator, GlobalDeduplicator

#: Default location for the gold dedup dataset when no ``dataset`` is passed.
_DEFAULT_DATASET = "benchmarks/data/dedup_gold.jsonl"

#: Fallback dedup knobs when a pipeline exposes no preprocessor config.
_DEFAULT_METHOD = "layered"
_DEFAULT_THRESHOLD = 0.85

#: Default sentence-encoder model used by semantic/layered dedup.
_DEFAULT_SEMANTIC_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

#: Process-wide encoder cache — the embedding model loads once per process.
_MODEL_CACHE: dict[str, Any] = {}


class DedupRunner:
    """Benchmark deduplication quality across pipeline instances.

    Each pipeline's dedup configuration (method, threshold) is whatever
    was set at construction time — the benchmark just runs and scores.

    Compares dedup decisions against gold duplicate/non-duplicate labels.
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

            print(DedupRunner.help())
        """
        return (
            "Document Deduplication Benchmark\n"
            "================================\n"
            "What it tests:\n"
            "  Whether a pipeline correctly identifies duplicate and\n"
            "  near-duplicate documents. Each pipeline's dedup method\n"
            "  (e.g., MinHash, semantic embeddings) is scored against\n"
            "  gold duplicate/non-duplicate pair labels.\n\n"
            "Why it matters:\n"
            "  Duplicate documents inflate entity counts and can bias\n"
            "  relation extraction. A good dedup step keeps the KG\n"
            "  compact and accurate without losing unique information.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"text_a": "...", "text_b": "...", "label": "duplicate"}\n'
            '  {"text_a": "...", "text_b": "...", "label": "not_duplicate"}\n\n'
            "Primary metrics:\n"
            "  Precision — fraction of predicted duplicates that are true duplicates.\n"
            "  Recall — fraction of true duplicates that were found.\n"
            "  F1 — harmonic mean of precision and recall.\n"
        )

    def run(self, pipelines: dict[str, Pipeline]) -> StageResult:
        """Run dedup benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.  Each pipeline should be
                fully configured (dedup method, threshold, etc.) but does
                not need ``input_paths`` or ``output_dir``.

        Returns:
            A ``StageResult`` wrapping ``{pipeline_name: {method, threshold,
            precision, recall, f1, runtime_seconds}}`` — one entry per
            pipeline, keyed by the name given in ``pipelines``.
        """
        Data.download("bench_dedup", path=str(self.dataset))
        gold = _load_gold(self.dataset)
        texts = list(dict.fromkeys(t for r in gold for t in (r["text_a"], r["text_b"])))

        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            method, threshold = _dedup_config(pipeline)
            t0 = time.perf_counter()
            y_true = [record["label"] for record in gold]
            if method == "exact":
                # Exact dedup merges identical content — O(pairs) decision.
                y_pred = [
                    "duplicate" if r["text_a"] == r["text_b"] else "not_duplicate" for r in gold
                ]
            elif method == "minhash":
                # One batched MinHash-LSH pass over all texts (matches how
                # the pipeline dedups a whole corpus) instead of per-pair runs.
                clusters = _dedup_clusters(texts, method, threshold)
                y_pred = [
                    "duplicate"
                    if any(r["text_a"] in c and r["text_b"] in c for c in clusters)
                    else "not_duplicate"
                    for r in gold
                ]
            else:
                y_pred = [_predict_pair(r["text_a"], r["text_b"], method, threshold) for r in gold]
            elapsed_s = time.perf_counter() - t0

            metrics = _score(y_true, y_pred)
            metrics["method"] = method
            metrics["threshold"] = threshold
            metrics["runtime_seconds"] = round(elapsed_s, 4)
            metrics["n_samples"] = len(gold)
            results[name] = metrics
        return StageResult(stage="dedup", results=results, dataset=self.dataset)


# ── Helpers ─────────────────────────────────────────────────────


def _load_gold(path: Path) -> list[dict[str, Any]]:
    """Load gold duplicate/non-duplicate pair records from a JSONL file.

    Args:
        path: Path to the gold JSONL file.

    Returns:
        List of ``{"text_a": str, "text_b": str, "label": "duplicate"|"not_duplicate"}``
        records.

    Raises:
        ValueError: If a record is malformed or uses an unknown label.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text_a" not in record or "text_b" not in record or "label" not in record:
                raise ValueError(
                    f"{path}:{line_no} — record must have 'text_a', 'text_b' and 'label' keys"
                )
            label = str(record["label"]).strip().lower()
            if label not in ("duplicate", "not_duplicate"):
                raise ValueError(
                    f"{path}:{line_no} — label must be 'duplicate' or 'not_duplicate', "
                    f"got {label!r}"
                )
            records.append(
                {
                    "text_a": str(record["text_a"]),
                    "text_b": str(record["text_b"]),
                    "label": label,
                }
            )
    if not records:
        raise ValueError(f"No gold records found in {path}")
    return records


def _dedup_config(pipeline: Pipeline) -> tuple[str, float]:
    """Read the dedup method/threshold from a pipeline's preprocessor config.

    Falls back to library defaults when the pipeline has no preprocessor or
    no ``config`` attribute (e.g. fully custom ``Pipeline`` subclasses).

    Args:
        pipeline: The pipeline instance to inspect.

    Returns:
        ``(method, threshold)`` used by the document dedup stage.
    """
    config = getattr(getattr(pipeline, "_preprocessor", None), "config", None)
    if config is None:
        return _DEFAULT_METHOD, _DEFAULT_THRESHOLD
    return config.doc_dedup_method, float(config.doc_dedup_threshold)


def _dedup_clusters(texts: list[str], method: str, threshold: float) -> list[set[str]]:
    """Cluster *texts* into duplicate groups in a single pass.

    Mirrors the pipeline's document dedup: ``"exact"`` merges identical
    content; ``"minhash"`` merges exact + MinHash-LSH near-duplicates.

    Args:
        texts: The unique texts to cluster.
        method: ``"exact"`` or ``"minhash"``.
        threshold: Similarity threshold (used by minhash).

    Returns:
        List of clusters, each a set of texts considered duplicates.

    Raises:
        ValueError: For unsupported methods.
    """
    if method == "exact":
        groups: dict[str, list[str]] = {}
        for text in texts:
            groups.setdefault(text, []).append(text)
        return [set(group) for group in groups.values() if len(group) > 1]
    if method == "minhash":
        records = [
            {"doc_id": str(index), "content": text, "quality_score": 0.0}
            for index, text in enumerate(texts)
        ]
        assignments = GlobalDeduplicator(threshold=threshold).cluster(records)
        clusters: dict[str, set[str]] = {}
        for index, text in enumerate(texts):
            cluster_id = assignments[str(index)].cluster_id
            if cluster_id:
                clusters.setdefault(cluster_id, set()).add(text)
        return list(clusters.values())
    raise ValueError(f"Unsupported batched dedup method: {method}")


def _predict_pair(text_a: str, text_b: str, method: str, threshold: float) -> str:
    """Predict whether two texts are duplicates under the pipeline's dedup method.

    Both texts are deduplicated together; if one is removed as a duplicate,
    the pair is predicted ``"duplicate"``, otherwise ``"not_duplicate"``.
    This mirrors what the pipeline's dedup stage does to the gold corpus.

    For embedding-based methods a process-cached encoder is injected so the
    model loads once instead of once per pair.
    """
    docs = [
        Document(content=text_a, doc_id="text_a"),
        Document(content=text_b, doc_id="text_b"),
    ]
    kwargs: dict[str, Any] = {"method": method, "threshold": threshold}
    if method in ("semantic", "layered"):
        kwargs["semantic_encoder"] = _semantic_encoder(_DEFAULT_SEMANTIC_MODEL)
    kept = Deduplicator(**kwargs).deduplicate(docs)
    return "duplicate" if len(kept) < len(docs) else "not_duplicate"


def _semantic_encoder(model_name: str):
    """Return a cached sentence-encoder callable for *model_name*.

    The embedding model is loaded once per process and shared across every
    pair/pipeline in the benchmark run.
    """
    encoder = _MODEL_CACHE.get(model_name)
    if encoder is None:
        encoder = _load_sentence_encoder(model_name)
        _MODEL_CACHE[model_name] = encoder
    return encoder


def _load_sentence_encoder(model_name: str):
    """Load a sentence-transformers model and return its ``encode`` callable."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name).encode


def _score(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Score duplicate predictions against gold labels.

    ``"duplicate"`` is treated as the positive class:

    * Accuracy — fraction of correct duplicate/not-duplicate decisions.
    * Precision — fraction of predicted duplicates that are true duplicates.
    * Recall — fraction of true duplicates that were found.
    * F1 — harmonic mean of precision and recall.
    """
    tp = sum(
        1 for t, p in zip(y_true, y_pred, strict=False) if t == "duplicate" and p == "duplicate"
    )
    fp = sum(
        1 for t, p in zip(y_true, y_pred, strict=False) if t == "not_duplicate" and p == "duplicate"
    )
    fn = sum(
        1 for t, p in zip(y_true, y_pred, strict=False) if t == "duplicate" and p == "not_duplicate"
    )
    tn = sum(
        1
        for t, p in zip(y_true, y_pred, strict=False)
        if t == "not_duplicate" and p == "not_duplicate"
    )
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
