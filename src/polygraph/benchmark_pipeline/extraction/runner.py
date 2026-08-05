"""Extraction benchmark — compare NER accuracy against gold entity annotations.

Each pipeline extracts entities from gold texts and span-level F1 is
computed against gold entity spans.

Gold dataset format (JSONL)::

    {"text": "...", "entities": [{"name": "IBM", "type": "ORG", "start": 0, "end": 3}]}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from polygraph._shared.stage_config import ExtractionConfig
from polygraph.data import Data
from polygraph.kg_build.extract._base import Entity, EntityExtractor
from polygraph.kg_build.extract.registry import create_entity_method
from polygraph.pipelines import Pipeline

logger = logging.getLogger(__name__)

#: Default location of the CoNLL-2003-derived NER gold dataset (see ``Data.download``).
_DEFAULT_DATASET = "benchmarks/data/ner_gold.jsonl"


class ExtractionRunner:
    """Benchmark entity extraction quality across pipeline instances.

    Compares extracted entity spans against gold annotations.
    Useful for evaluating different NER backends (spaCy, LLM-based, etc.).
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

            print(ExtractionRunner.help())
        """
        return (
            "Entity Extraction Benchmark\n"
            "============================\n"
            "What it tests:\n"
            "  Whether a pipeline correctly identifies entity spans\n"
            "  (name, type, position) in raw text. Compares extracted\n"
            "  entities against gold-standard annotations.\n\n"
            "Why it matters:\n"
            "  Entity extraction is the foundation of any KG. Missed\n"
            "  entities create gaps; spurious entities add noise.\n"
            "  Different backends (spaCy, LLM prompts, rule-based)\n"
            "  have different strengths and failure modes.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"text": "...", "entities": [{"name": "IBM", "type": "ORG", "start": 0, "end": 3}]}\n\n'
            "Primary metrics:\n"
            "  Span-level precision — how many extracted spans are correct.\n"
            "  Span-level recall — how many gold spans were found.\n"
            "  Span-level F1 — harmonic mean of precision and recall.\n"
            "  Entity-type accuracy — whether the entity type (PER/ORG/LOC/etc.) is correct.\n"
        )

    def run(self, pipelines: dict[str, Pipeline]) -> dict[str, Any]:
        """Run extraction benchmark and return metrics per pipeline."""
        # Ensure the gold dataset is present (cached download if missing).
        Data.download("bench_ner", path=str(self.dataset))

        gold_records = _load_gold(self.dataset)
        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            try:
                results[name] = _benchmark_pipeline(pipeline, gold_records)
            except Exception as exc:
                logger.warning("Extraction benchmark failed for pipeline %r: %s", name, exc)
                results[name] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "type_accuracy": 0.0,
                    "runtime_seconds": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return results


# ── Helpers ───────────────────────────────────────────────────


def _load_gold(path: Path) -> list[dict[str, Any]]:
    """Load gold NER records from a JSONL file.

    Each record is ``{"text": ..., "entities": [{"name", "type", "start", "end"}]}``.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extraction_config(pipeline: Pipeline) -> ExtractionConfig:
    """Resolve the effective extraction config for a pipeline.

    Prefers the typed ``pipeline.extraction`` attribute set at construction
    time; falls back to the legacy raw ``extraction`` dict in ``_config``.
    """
    extraction = getattr(pipeline, "extraction", None)
    if isinstance(extraction, dict):
        extraction = ExtractionConfig.from_dict(extraction)
    if extraction is None:
        extraction = ExtractionConfig.from_dict(pipeline._config.get("extraction"))
    return extraction


def _entity_extractor(config: ExtractionConfig) -> EntityExtractor:
    """Build the entity extractor for an extraction config via the registry.

    Mirrors the pipeline default so the benchmark measures what the pipeline
    would actually run: spaCy falls back to ``en_core_web_sm`` when no model
    is configured explicitly.
    """
    entity_options = dict(config.entity_options or {})
    if config.entity_method == "spacy" and "model_name" not in entity_options:
        entity_options["model_name"] = "en_core_web_sm"
    return create_entity_method(config.entity_method, **entity_options)


def _gold_spans(record: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    """Normalize a gold record's entities into ``(start, end, type, name)`` tuples."""
    spans: list[tuple[int, int, str, str]] = []
    for ent in record.get("entities", []):
        start = int(ent.get("start", -1))
        end = int(ent.get("end", -1))
        if start < 0 or end < start:
            continue
        spans.append((start, end, str(ent.get("type", "")), str(ent.get("name", ""))))
    return spans


def _occurrences(name: str, text: str) -> list[tuple[int, int]]:
    """Return every ``(start, end)`` occurrence of *name* in *text*."""
    occurrences: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(name, start)
        if idx == -1:
            break
        occurrences.append((idx, idx + len(name)))
        start = idx + 1
    return occurrences


def _resolve_spans(
    entities: list[Entity],
    text: str,
    gold_spans: set[tuple[int, int]],
) -> list[tuple[int, int, str, str]]:
    """Resolve extracted entity names to ``(start, end, label, name)`` spans.

    Entity extractors return names, not offsets, so each name is located in
    the text. Occurrences that line up with a gold span are preferred; the
    first remaining unclaimed occurrence is used otherwise. Names that cannot
    be found in the text are skipped.
    """
    spans: list[tuple[int, int, str, str]] = []
    claimed: set[tuple[int, int]] = set()

    for entity in entities:
        name = entity.name
        if not name:
            continue
        occurrences = _occurrences(name, text)
        chosen: tuple[int, int] | None = None
        for occ in occurrences:
            if occ in gold_spans and occ not in claimed:
                chosen = occ
                break
        if chosen is None:
            for occ in occurrences:
                if occ not in claimed:
                    chosen = occ
                    break
        if chosen is None:
            continue
        claimed.add(chosen)
        spans.append((chosen[0], chosen[1], entity.label, name))

    return spans


def _span_metrics(
    gold: list[tuple[int, int, str, str]],
    predicted: list[tuple[int, int, str, str]],
) -> dict[str, float]:
    """Compute span-level precision / recall / F1 (CoNLL-style micro-average).

    A prediction is a true positive when its ``(start, end)`` matches a gold
    span. ``type_accuracy`` reports how often matched spans also agree on the
    entity type.
    """
    gold_by_span = {(g[0], g[1]): g for g in gold}
    matched_gold: set[tuple[int, int]] = set()
    tp = 0
    type_correct = 0

    for start, end, label, _name in predicted:
        if (start, end) in gold_by_span and (start, end) not in matched_gold:
            tp += 1
            matched_gold.add((start, end))
            if label == gold_by_span[(start, end)][2]:
                type_correct += 1

    fp = len(predicted) - tp
    fn = len(gold) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    type_accuracy = type_correct / tp if tp else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "type_accuracy": round(type_accuracy, 4),
    }


def _benchmark_pipeline(
    pipeline: Pipeline,
    gold_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one pipeline's entity extraction over the gold corpus and score it."""
    config = _extraction_config(pipeline)
    extractor = _entity_extractor(config)

    t0 = time.perf_counter()
    all_gold: list[tuple[int, int, str, str]] = []
    all_predicted: list[tuple[int, int, str, str]] = []
    for record in gold_records:
        text = str(record.get("text", ""))
        gold = _gold_spans(record)
        all_gold.extend(gold)
        all_predicted.extend(
            _resolve_spans(extractor.extract(text), text, {(g[0], g[1]) for g in gold})
        )
    elapsed_s = time.perf_counter() - t0

    metrics = _span_metrics(all_gold, all_predicted)
    metrics["runtime_seconds"] = round(elapsed_s, 4)
    metrics["entity_method"] = config.entity_method
    metrics["mode"] = config.mode
    return metrics
