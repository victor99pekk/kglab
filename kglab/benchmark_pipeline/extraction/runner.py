"""Extraction benchmark — compare NER accuracy against gold entity annotations.

Each pipeline extracts entities from gold texts and span-level F1 is
computed against gold entity spans.

Gold dataset format (JSONL)::

    {"text": "...", "entities": [{"name": "IBM", "type": "ORG", "start": 0, "end": 3}]}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kglab._shared.stage_config import ExtractionConfig
from kglab.benchmark_pipeline._bundled import bundled_data_dir
from kglab.benchmark_pipeline.report import StageResult
from kglab.data import Data
from kglab.kg_build.extract._base import Entity, EntityExtractor
from kglab.kg_build.extract.registry import create_entity_method
from kglab.pipelines import Pipeline

#: Default location of the CoNLL-2003-derived NER gold dataset — the file
#: bundled with the library (see also ``Data.download`` for other golds).
_DEFAULT_DATASET = str(bundled_data_dir() / "ner_gold.jsonl")

#: Gold-side label normalization — reconciles common label-name variants
#: (e.g. spaCy's OntoNotes ``"PERSON"`` vs CoNLL ``"PER"``) to a canonical
#: set before scoring. Defined once here, not per extractor: new extractors
#: need nothing; each gold's own label set decides what is scored.
NER_LABEL_ALIASES: dict[str, str] = {
    "PERSON": "PER",
    "GPE": "LOC",
    "LOC": "LOC",
    "NORP": "MISC",
    "ORG": "ORG",
    "FAC": "MISC",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
}

#: Available extraction gold datasets — pick the one matching your extractor's
#: schema. Keys are ``Data.download`` names; prefer TEST splits for evaluation.
EXTRACTION_GOLDS: dict[str, dict[str, Any]] = {
    "bench_ner": {
        "description": "CoNLL-2003 NER — TRAIN split (in-library default; not a held-out evaluation).",
        "types": "PER / LOC / ORG / MISC",
        "fits": "extractors emitting PER/LOC/ORG/MISC (e.g. BERT-NER, Flair CoNLL)",
    },
    "bench_ner_test": {
        "description": "CoNLL-2003 NER — TEST split (held-out; what published numbers use).",
        "types": "PER / LOC / ORG / MISC",
        "fits": "extractors emitting PER/LOC/ORG/MISC",
    },
    "bench_ner_wikiann": {
        "description": "wikiann (English) NER — Wikipedia-derived, TEST split.",
        "types": "PER / ORG / LOC",
        "fits": "broad 3-type extractors; spaCy PERSON/GPE/LOC/ORG via aliases",
    },
    "bench_ner_fewnerd": {
        "description": "FewNERD NER — coarse fine-grained types, TEST split (gated on HF: accept license + set HF_TOKEN).",
        "types": "PER / ORG / LOC / ART / BUILDING / EVENT / PRODUCT / OTHER",
        "fits": "fine-grained extractors (coarse-level comparison)",
    },
}


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

    @classmethod
    def golds(cls) -> str:
        """List the available extraction gold datasets and which extractor each fits.

        Each entry is a ``Data.download`` key: fetch it with
        ``Data.download("<key>", path=...)`` and pass the path as ``dataset=``.
        Prefer the TEST-split golds for real evaluation — training-split golds
        are not held-out.
        """
        lines = [
            "Extraction gold datasets — pick the one matching your extractor:",
            "  fetch: Data.download('<key>', path=...)   then   Benchmark.Extraction(dataset=path)",
        ]
        for key, info in EXTRACTION_GOLDS.items():
            lines.append(f"\n  {key}")
            lines.append(f"    {info['description']}")
            lines.append(f"    types: {info['types']}")
            lines.append(f"    fits:  {info['fits']}")
        return "\n".join(lines)

    def run(
        self,
        pipelines: dict[str, Pipeline],
        *,
        max_records: int | None = None,
    ) -> StageResult:
        """Run extraction benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.
            max_records: Cap the number of gold records scored to the
                first *N* (``None`` scores all of them).  The bundled
                CoNLL gold has ~14k records; capping keeps a demo fast
                while still scoring a slice of the bundled gold.

        Returns:
            A ``StageResult`` wrapping ``{pipeline_name: {precision, recall,
            f1, type_accuracy, runtime_seconds, entity_method, mode}}`` — one
            entry per pipeline, keyed by the name given in ``pipelines``.
        """
        # Ensure the gold dataset is present (cached download if missing).
        Data.download("bench_ner", path=str(self.dataset))

        gold_records = _load_gold(self.dataset)
        if max_records is not None:
            gold_records = gold_records[:max_records]
        # The gold's own label set defines the schema — predictions whose
        # (aliased) label is not one of these are outside the gold and not scored.
        allowed_types = {g[2] for record in gold_records for g in _gold_spans(record)}
        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            results[name] = _benchmark_pipeline(pipeline, gold_records, allowed_types)
        return StageResult(stage="extraction", results=results, dataset=self.dataset)


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


def _count_record(
    gold: list[tuple[int, int, str, str]],
    predicted: list[tuple[int, int, str, str]],
    allowed_types: set[str],
    aliases: dict[str, str] | None = None,
) -> dict[str, int]:
    """Count TP/FP/FN/type matches for ONE record's gold and predicted spans.

    Only predictions whose (aliased) label is in the gold's label set are
    scored — labels outside it (noun-phrase ``CONCEPT``s, dates, ...) are
    neither true nor false positives. A span match counts as a true positive
    even when the type differs; only ``type_accuracy`` penalizes that.

    Spans are record-relative — the same ``(start, end)`` recurs across
    records (e.g. many articles start with "EU") — so matching happens per
    record; callers aggregate counts with ``_aggregate_metrics``.
    """
    aliases = aliases or {}
    gold_by_span = {(g[0], g[1]): g for g in gold}
    matched_gold: set[tuple[int, int]] = set()
    tp = 0
    type_correct = 0
    n_scored = 0

    for start, end, label, _name in predicted:
        mapped = aliases.get(label, label)
        if mapped not in allowed_types:
            continue
        n_scored += 1
        if (start, end) in gold_by_span and (start, end) not in matched_gold:
            tp += 1
            matched_gold.add((start, end))
            if mapped == gold_by_span[(start, end)][2]:
                type_correct += 1

    return {
        "tp": tp,
        "fp": n_scored - tp,
        "fn": len(gold) - len(matched_gold),
        "type_correct": type_correct,
        "n_predicted": len(predicted),
        "n_scored": n_scored,
        "n_dropped": len(predicted) - n_scored,
    }


def _aggregate_metrics(records: list[dict[str, int]]) -> dict[str, Any]:
    """Sum per-record counts into micro-averaged P/R/F1 + transparency keys."""
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    type_correct = sum(r["type_correct"] for r in records)
    n_predicted = sum(r["n_predicted"] for r in records)
    n_scored = sum(r["n_scored"] for r in records)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    type_accuracy = type_correct / tp if tp else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "type_accuracy": round(type_accuracy, 4),
        "n_predicted": n_predicted,
        "n_scored": n_scored,
        "n_dropped": n_predicted - n_scored,
    }


def _span_metrics(
    gold: list[tuple[int, int, str, str]],
    predicted: list[tuple[int, int, str, str]],
    allowed_types: set[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Metrics for a single record (convenience; used by tests)."""
    allowed = allowed_types or {g[2] for g in gold}
    return _aggregate_metrics([_count_record(gold, predicted, allowed, aliases)])


def _benchmark_pipeline(
    pipeline: Pipeline,
    gold_records: list[dict[str, Any]],
    allowed_types: set[str],
) -> dict[str, Any]:
    """Run one pipeline's entity extraction over the gold corpus and score it."""
    config = _extraction_config(pipeline)
    extractor = _entity_extractor(config)

    t0 = time.perf_counter()
    counts: list[dict[str, int]] = []
    for record in gold_records:
        text = str(record.get("text", ""))
        gold = _gold_spans(record)
        predicted = _resolve_spans(extractor.extract(text), text, {(g[0], g[1]) for g in gold})
        counts.append(_count_record(gold, predicted, allowed_types, NER_LABEL_ALIASES))
    elapsed_s = time.perf_counter() - t0

    metrics = _aggregate_metrics(counts)
    metrics["runtime_seconds"] = round(elapsed_s, 4)
    metrics["entity_method"] = config.entity_method
    metrics["mode"] = config.mode
    return metrics
