"""Chunking benchmark — compare chunk boundary accuracy against gold splits.

Each pipeline chunks the gold texts and boundary F1 is computed against
gold chunk boundaries.

Gold dataset format (JSONL)::

    {"text": "...", "chunks": [{"name": "EU", "type": "ORG"}, ...]}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from kglab._shared import Document
from kglab.benchmark_pipeline._bundled import bundled_data_dir
from kglab.benchmark_pipeline.report import StageResult
from kglab.data import Data
from kglab.pipelines import Pipeline
from kglab.preprocess import chunk
from kglab.preprocess.chunk import SemanticChunker

#: Default location for the gold chunking dataset when no ``dataset`` is passed —
#: the file bundled with the library (no cwd or repo-layout assumptions).
DEFAULT_CHUNKING_DATASET = str(bundled_data_dir() / "chunking_gold.jsonl")

#: Human-readable description of the gold dataset schema.
DATASET_FORMAT = """\
Each line of the gold JSONL is one object:
  "text"   — raw text to chunk.
  "chunks" — gold entities that must stay intact within a single chunk,
             as a list of {"name": str, "type": str}.
"""

#: Fallback chunking knobs when a pipeline exposes no preprocessing summary.
_DEFAULT_CHUNK_METHOD = "sentence"
_DEFAULT_TARGET_TOKENS = 450
_DEFAULT_OVERLAP_TOKENS = 60
_DEFAULT_SEMANTIC_THRESHOLD = 0.55
_DEFAULT_SEMANTIC_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class ChunkingRunner:
    """Benchmark chunking quality across pipeline instances.

    Compares the chunk boundaries produced by each pipeline against
    a gold-standard segmentation.  Useful for evaluating different
    chunking strategies (sentence, semantic, fixed-size, etc.).

    Primary metric: boundary F1 (how well chunk boundaries align with gold).
    """

    def __init__(self, dataset: str | Path | None = None) -> None:
        self.dataset = Path(dataset) if dataset is not None else None

    @classmethod
    def help(cls) -> str:
        """Return a human-readable description of this benchmark.

        Use this to understand what the benchmark measures, the gold
        dataset format, and the metrics it produces — without reading
        the source code.

        Example::

            print(ChunkingRunner.help())
        """
        return (
            "Chunking Benchmark\n"
            "==================\n"
            "What it tests:\n"
            "  Whether a pipeline splits text into chunks that align with\n"
            "  a gold-standard segmentation. Different chunking strategies\n"
            "  (sentence, semantic, fixed-size) can dramatically affect\n"
            "  downstream entity extraction and relation quality.\n\n"
            "Why it matters:\n"
            "  Chunk boundaries determine what context an entity extractor\n"
            "  sees. Poor chunking can split entities across chunks or\n"
            "  merge unrelated concepts, degrading KG quality.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"text": "...", "chunks": ["chunk 1", "chunk 2", ...]}\n\n'
            "Primary metric:\n"
            "  Boundary F1 — harmonic mean of precision and recall of\n"
            "  chunk boundaries against gold boundaries.\n"
        )

    def run(
        self,
        pipelines: dict[str, Pipeline],
        *,
        max_records: int | None = None,
    ) -> StageResult:
        """Run chunking benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.  Each pipeline should be
                fully configured (chunk method, target tokens, etc.) but
                does not need ``input_paths`` or ``output_dir``.
            max_records: Cap the number of gold records scored to the
                first *N* (``None`` scores all of them).  Lets a demo use
                a slice of the bundled gold without creating custom data.

        Returns:
            A ``StageResult`` wrapping ``{pipeline_name: {method, precision,
            recall, f1, runtime_seconds}}`` — one entry per pipeline, keyed
            by the name given in ``pipelines``.
        """
        dataset = _resolve_dataset(self.dataset)
        gold = _load_gold(dataset)
        if max_records is not None:
            gold = gold[:max_records]

        results: dict[str, Any] = {}
        for name, pipeline in pipelines.items():
            method, options = _chunk_config(pipeline)
            t0 = time.perf_counter()
            predicted = _chunk_records(gold, method, options)
            elapsed_s = time.perf_counter() - t0

            metrics = _score_all(gold, predicted)
            metrics["method"] = method
            metrics["runtime_seconds"] = round(elapsed_s, 4)
            metrics["n_samples"] = len(gold)
            metrics["target_tokens"] = options.get(
                "target_tokens", options.get("size", _DEFAULT_TARGET_TOKENS)
            )
            metrics["overlap_tokens"] = options.get(
                "overlap_tokens", options.get("overlap", _DEFAULT_OVERLAP_TOKENS)
            )
            results[name] = metrics
        return StageResult(stage="chunking", results=results, dataset=dataset)


# ── Helpers ─────────────────────────────────────────────────────


def _resolve_dataset(dataset: Path | None) -> Path:
    """Resolve the gold dataset path, downloading it if missing.

    When ``dataset`` is ``None`` (the default), the gold bundled with the
    library is used; ``Data.download`` only fetches it if it is not present.
    """
    target = dataset if dataset is not None else Path(DEFAULT_CHUNKING_DATASET)
    Data.download("bench_chunking", path=str(target))
    return target


def _load_gold(path: Path) -> list[dict[str, Any]]:
    """Load gold chunking records from a JSONL file.

    Each record is ``{"text": str, "chunks": [{"name": str, "type": str}, ...]}``
    where ``chunks`` lists gold entities that must stay intact within a
    single chunk.

    Args:
        path: Path to the gold JSONL file.

    Returns:
        List of ``{"text": str, "chunks": list[dict]}`` records.

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
            if "text" not in record or "chunks" not in record:
                raise ValueError(f"{path}:{line_no} — record must have 'text' and 'chunks' keys")
            chunks = record["chunks"]
            if not isinstance(chunks, list):
                raise ValueError(f"{path}:{line_no} — 'chunks' must be a list")
            records.append({"text": str(record["text"]), "chunks": chunks})
    if not records:
        raise ValueError(f"No gold records found in {path}")
    return records


def _chunk_config(pipeline: Pipeline) -> tuple[str, dict[str, Any]]:
    """Read chunk method + options from a pipeline's preprocessing knobs.

    Pipelines that expose a ``preprocess_summary`` dict (e.g. ``Baseline``)
    report their chunk settings; custom pipelines that override
    ``preprocess()`` have no summary and fall back to library defaults.

    Args:
        pipeline: The pipeline instance to inspect.

    Returns:
        ``(method, options)`` where ``options`` is the kwargs dict for the
        chunk registry function.
    """
    summary = getattr(pipeline, "preprocess_summary", None)
    if summary is None:
        return _DEFAULT_CHUNK_METHOD, {
            "target_tokens": _DEFAULT_TARGET_TOKENS,
            "overlap_tokens": _DEFAULT_OVERLAP_TOKENS,
            "threshold": _DEFAULT_SEMANTIC_THRESHOLD,
            "model": _DEFAULT_SEMANTIC_MODEL,
        }
    return summary.get("chunk_method", _DEFAULT_CHUNK_METHOD), {
        "target_tokens": summary.get("chunk_target_tokens", _DEFAULT_TARGET_TOKENS),
        "overlap_tokens": summary.get("chunk_overlap_tokens", _DEFAULT_OVERLAP_TOKENS),
        "threshold": summary.get("chunk_semantic_threshold", _DEFAULT_SEMANTIC_THRESHOLD),
        "model": summary.get("chunk_semantic_model", _DEFAULT_SEMANTIC_MODEL),
    }


def _chunk_records(
    gold: list[dict[str, Any]],
    method: str,
    options: dict[str, Any],
) -> list[list[str]]:
    """Chunk every gold text with the given method, returning chunk contents.

    For ``semantic`` chunking a single model-backed chunker is reused across
    all records so the embedding model loads once per run instead of once per
    record.
    """
    if method == "semantic":
        chunker = SemanticChunker(
            target_tokens=options.get("target_tokens", _DEFAULT_TARGET_TOKENS),
            overlap_tokens=options.get("overlap_tokens", _DEFAULT_OVERLAP_TOKENS),
            similarity_threshold=options.get("threshold", _DEFAULT_SEMANTIC_THRESHOLD),
            model_name=options.get("model", _DEFAULT_SEMANTIC_MODEL),
        )
        return [
            [c.content for c in chunker.chunk([Document(content=record["text"])])]
            for record in gold
        ]
    return [_chunk_text(record["text"], method, options) for record in gold]


def _chunk_text(text: str, method: str, options: dict[str, Any]) -> list[str]:
    """Run one gold text through the pipeline's chunk method.

    Uses the same ``kglab.preprocess.chunk`` registry as
    ``Baseline.preprocess`` so the benchmark measures what the pipeline
    would actually produce.
    """
    docs = [Document(content=text)]
    if method == "sentence":
        chunked = chunk.by_sentence(
            docs,
            target_tokens=options.get("target_tokens", _DEFAULT_TARGET_TOKENS),
            overlap_tokens=options.get("overlap_tokens", _DEFAULT_OVERLAP_TOKENS),
        )
    elif method == "fixed":
        chunked = chunk.by_fixed(
            docs,
            size=options.get("size", options.get("target_tokens", _DEFAULT_TARGET_TOKENS)),
            overlap=options.get("overlap", options.get("overlap_tokens", _DEFAULT_OVERLAP_TOKENS)),
        )
    elif method == "semantic":
        chunked = chunk.by_semantic(
            docs,
            target_tokens=options.get("target_tokens", _DEFAULT_TARGET_TOKENS),
            overlap_tokens=options.get("overlap_tokens", _DEFAULT_OVERLAP_TOKENS),
            threshold=options.get("threshold", _DEFAULT_SEMANTIC_THRESHOLD),
            model=options.get("model", _DEFAULT_SEMANTIC_MODEL),
        )
    else:
        raise ValueError(f"Unknown chunk method '{method}'. Available: sentence, fixed, semantic")
    return [c.content for c in chunked]


def _score_all(
    gold: list[dict[str, Any]],
    predicted: list[list[str]],
) -> dict[str, float]:
    """Micro-average boundary F1 across all gold records.

    * Precision — fraction of chunk boundaries that do NOT fall strictly
      inside a gold entity span (``1.0`` when there are no boundaries).
    * Recall — fraction of gold entities with NO boundary strictly inside
      them (``1.0`` when no entities are locatable).
    * F1 — harmonic mean of the two.
    """
    total_boundaries = 0
    good_boundaries = 0
    total_entities = 0
    intact_entities = 0

    for record, chunk_texts in zip(gold, predicted, strict=False):
        text = _normalize(record["text"])
        entity_spans = _entity_spans(text, record["chunks"])
        boundaries = _chunk_boundaries(text, chunk_texts)

        total_boundaries += len(boundaries)
        good_boundaries += sum(1 for b in boundaries if not any(s < b < e for s, e in entity_spans))
        total_entities += len(entity_spans)
        intact_entities += sum(1 for s, e in entity_spans if not any(s < b < e for b in boundaries))

    precision = good_boundaries / total_boundaries if total_boundaries else 1.0
    recall = intact_entities / total_entities if total_entities else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _normalize(text: str) -> str:
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _entity_spans(
    text: str,
    entities: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Locate gold entity names in the normalized text as ``(start, end)``.

    Entities that cannot be located in the text are skipped (they are not
    counted against recall — the benchmark only scores locatable entities).
    """
    spans: list[tuple[int, int]] = []
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        if not name:
            continue
        span = _find_normalized(text, name)
        if span is not None:
            spans.append(span)
    return spans


def _find_normalized(text: str, needle: str) -> tuple[int, int] | None:
    """Locate *needle* in whitespace-normalized *text*.

    Tries an exact match first, then a case-insensitive fallback.  Returns
    ``None`` when the needle cannot be found.
    """
    norm_needle = _normalize(needle)
    if not norm_needle:
        return None
    start = text.find(norm_needle)
    if start == -1:
        start = text.casefold().find(norm_needle.casefold())
        if start == -1:
            return None
    return start, start + len(norm_needle)


def _chunk_boundaries(text: str, chunk_texts: list[str]) -> list[int]:
    """Align predicted chunks to *text* and return boundary end offsets.

    Chunks are greedily aligned with ``str.find``, searching from the
    previous chunk's start so overlapping chunks still align.  The
    boundaries are the end offsets of every aligned chunk except the last
    one — i.e. where one chunk ends and the next begins.
    """
    boundaries: list[int] = []
    search_from = 0
    for index, chunk_text in enumerate(chunk_texts):
        norm_chunk = _normalize(chunk_text)
        if not norm_chunk:
            continue
        start = text.find(norm_chunk, search_from)
        if start == -1:
            start = text.casefold().find(norm_chunk.casefold(), search_from)
            if start == -1:
                # Unaligned chunk — contributes no boundary.
                continue
        end = start + len(norm_chunk)
        search_from = start
        if index < len(chunk_texts) - 1:
            boundaries.append(end)
    return boundaries
