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

from polygraph._shared import Document
from polygraph.data import Data
from polygraph.pipelines import Pipeline
from polygraph.preprocess.dedup import Deduplicator

#: Default location for the gold dedup dataset when no ``dataset`` is passed.
_DEFAULT_DATASET = "benchmarks/data/dedup_gold.jsonl"

#: Fallback dedup knobs when a pipeline exposes no preprocessor config.
_DEFAULT_METHOD = "layered"
_DEFAULT_THRESHOLD = 0.85


class DedupRunner:
    """Benchmark deduplication quality across pipeline instances.

    Each pipeline's dedup configuration (method, threshold) is whatever
    was set at construction time — the benchmark just runs and scores.

    Compares dedup decisions against gold duplicate/non-duplicate labels.
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

    def run(self, pipelines: dict[str, Pipeline]) -> dict[str, Any]:
        """Run dedup benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.  Each pipeline should be
                fully configured (dedup method, threshold, etc.) but does
                not need ``input_paths`` or ``output_dir``.

        Returns:
            ``{pipeline_name: {method, threshold, precision, recall, f1,
            runtime_seconds}}`` — one entry per pipeline, keyed by the
            name given in ``pipelines``.
        """
        Data.download("bench_dedup", path=str(self.dataset))
        gold = _load_gold(self.dataset)

        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            try:
                method, threshold = _dedup_config(pipeline)
                t0 = time.perf_counter()
                y_true = [record["label"] for record in gold]
                y_pred = [
                    _predict_pair(record["text_a"], record["text_b"], method, threshold)
                    for record in gold
                ]
                elapsed_s = time.perf_counter() - t0

                metrics = _score(y_true, y_pred)
                metrics["method"] = method
                metrics["threshold"] = threshold
                metrics["runtime_seconds"] = round(elapsed_s, 4)
                metrics["n_samples"] = len(gold)
                results[name] = metrics
            except Exception as exc:
                results[name] = {
                    "method": None,
                    "threshold": None,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "runtime_seconds": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return results


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


def _predict_pair(text_a: str, text_b: str, method: str, threshold: float) -> str:
    """Predict whether two texts are duplicates under the pipeline's dedup method.

    Both texts are deduplicated together; if one is removed as a duplicate,
    the pair is predicted ``"duplicate"``, otherwise ``"not_duplicate"``.
    This mirrors what the pipeline's dedup stage does to the gold corpus.
    """
    docs = [
        Document(content=text_a, doc_id="text_a"),
        Document(content=text_b, doc_id="text_b"),
    ]
    kept = Deduplicator(method=method, threshold=threshold).deduplicate(docs)
    return "duplicate" if len(kept) < len(docs) else "not_duplicate"


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
