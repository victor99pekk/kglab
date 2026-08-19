"""Quality filter benchmark — compare garbage detection against gold labels.

Each pipeline's quality filter thresholds are tested against gold keep/reject
labels.

Gold dataset format (JSONL)::

    {"text": "...", "label": "keep"}
    {"text": "...", "label": "reject"}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kglab.benchmark_pipeline.report import StageResult
from kglab.data import Data
from kglab.pipelines import Pipeline
from kglab.preprocess.quality import QualityProfiler, QualityThresholds

#: Gold dataset format (JSONL) — one record per line.
DATASET_FORMAT = (
    '{"text": "...", "label": "keep"|"reject"}  '
    "— each record pairs a raw text with the ground-truth quality decision."
)

#: Default location for the gold quality dataset when no ``dataset`` is passed.
_DEFAULT_DATASET = "benchmarks/data/quality_gold.jsonl"

#: Default quality thresholds used when a pipeline exposes no preprocessor config.
_DEFAULT_MIN_CHARS = 200
_DEFAULT_MIN_WORDS = 40


class QualityFilterRunner:
    """Benchmark quality filter accuracy across pipeline instances.

    Tests whether a pipeline's quality thresholds correctly separate
    useful content from garbage (boilerplate, ads, low-information text).
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

            print(QualityFilterRunner.help())
        """
        return (
            "Quality Filter Benchmark\n"
            "========================\n"
            "What it tests:\n"
            "  Whether a pipeline's quality filter correctly keeps\n"
            "  informative content and rejects low-quality text\n"
            "  (boilerplate, navigation, ads, noise).\n\n"
            "Why it matters:\n"
            "  Low-quality text pollutes the KG with spurious entities\n"
            "  and nonsensical relations. Overly aggressive filtering\n"
            "  discards valid content. The right balance preserves\n"
            "  recall while maintaining precision.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"text": "...", "label": "keep"}\n'
            '  {"text": "...", "label": "reject"}\n\n'
            "Primary metrics:\n"
            "  Accuracy — overall correct keep/reject decisions.\n"
            "  Precision — fraction of kept texts that should be kept.\n"
            "  Recall — fraction of good texts that were kept.\n"
        )

    def run(self, pipelines: dict[str, Pipeline]) -> StageResult:
        """Run quality filter benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.  Each pipeline should be
                fully configured (quality thresholds, etc.) but does not
                need ``input_paths`` or ``output_dir``.

        Returns:
            A ``StageResult`` wrapping ``{name: metrics}`` where metrics
            holds ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``runtime_seconds`` and the thresholds used.  A pipeline that
            fails raises — failures are never silently reported as zeroed
            metrics.
        """
        # Ensure the gold dataset is available. ``bench_quality`` (TACRED)
        # is license-gated — ``Data.download`` raises ``RuntimeError`` until
        # a gold JSONL is manually placed at ``self.dataset``.
        Data.download("bench_quality", path=self.dataset)

        gold = _load_gold(self.dataset)

        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            min_chars, min_words = _extract_thresholds(pipeline)

            t0 = time.perf_counter()
            y_true = [record["label"] for record in gold]
            y_pred = [_predict(record["text"], min_chars, min_words) for record in gold]
            elapsed_s = time.perf_counter() - t0

            metrics = _score(y_true, y_pred)
            metrics["runtime_seconds"] = round(elapsed_s, 4)
            metrics["n_samples"] = len(gold)
            metrics["quality_min_chars"] = min_chars
            metrics["quality_min_words"] = min_words
            results[name] = metrics
        return StageResult(stage="quality", results=results, dataset=self.dataset)


# ── Helpers ─────────────────────────────────────────────────────


def _load_gold(dataset: Path) -> list[dict[str, Any]]:
    """Load gold keep/reject records from a JSONL file.

    Args:
        dataset: Path to the gold JSONL file.

    Returns:
        List of ``{"text": str, "label": "keep" | "reject"}`` records.

    Raises:
        ValueError: If a record is malformed, uses an unknown label, or
            the file contains no records.
    """
    records: list[dict[str, Any]] = []
    with dataset.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" not in record or "label" not in record:
                raise ValueError(f"{dataset}:{line_no} — record must have 'text' and 'label' keys")
            label = str(record["label"]).strip().lower()
            if label not in ("keep", "reject"):
                raise ValueError(
                    f"{dataset}:{line_no} — label must be 'keep' or 'reject', got {label!r}"
                )
            records.append({"text": str(record["text"]), "label": label})
    if not records:
        raise ValueError(f"No gold records found in {dataset}")
    return records


def _extract_thresholds(pipeline: Pipeline) -> tuple[int, int]:
    """Read quality thresholds from a pipeline's preprocessor config.

    Falls back to library defaults when the pipeline has no preprocessor
    or no ``config`` attribute (e.g. fully custom ``Pipeline`` subclasses).

    Args:
        pipeline: The pipeline instance to inspect.

    Returns:
        ``(min_chars, min_words)`` thresholds for the quality filter.
    """
    preprocessor = getattr(pipeline, "_preprocessor", None)
    config = getattr(preprocessor, "config", None)
    if config is None:
        return _DEFAULT_MIN_CHARS, _DEFAULT_MIN_WORDS
    return int(config.quality_min_chars), int(config.quality_min_words)


def _predict(text: str, min_chars: int, min_words: int) -> str:
    """Predict ``"keep"`` or ``"reject"`` for a single text.

    Uses ``QualityProfiler`` with the pipeline's thresholds — the same
    decision logic as the preprocess ``quality`` stage.

    Args:
        text: Raw text to evaluate.
        min_chars: Minimum character count to keep a text.
        min_words: Minimum word count to keep a text.

    Returns:
        ``"keep"`` when the text passes the quality profile, else ``"reject"``.
    """
    profiler = QualityProfiler(
        thresholds=QualityThresholds(min_chars=min_chars, min_words=min_words)
    )
    profile = profiler.profile(text)
    return "keep" if profile.accepted else "reject"


def _score(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Score keep/reject predictions against gold labels.

    ``"keep"`` is treated as the positive class:

    * Accuracy — fraction of correct keep/reject decisions.
    * Precision — fraction of kept texts that should be kept.
    * Recall — fraction of good texts that were kept.
    * F1 — harmonic mean of precision and recall.

    Args:
        y_true: Gold labels (``"keep"`` / ``"reject"``).
        y_pred: Predicted labels (``"keep"`` / ``"reject"``).

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, and ``f1``.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == "keep" and p == "keep")
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == "reject" and p == "keep")
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == "keep" and p == "reject")
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == "reject" and p == "reject")
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
