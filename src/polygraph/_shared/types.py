"""Shared types used across all pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single document with metadata — the universal currency between stages.

    Every pipeline stage accepts and/or returns ``Document`` instances. A
    ``Document`` can represent a raw source file, a cleaned text, or a chunk
    — the ``metadata`` dict carries stage-specific annotations (chunk index,
    parent document ID, token count, etc.).

    Attributes:
        content: The full text body of the document or chunk.
        source: Origin path, URL, or identifier.
        doc_id: Stable unique identifier. Used for deduplication and
            provenance tracking across pipeline runs.
        language: ISO 639-1 language code (e.g. ``\"en\"``, ``\"vi\"``).
            Defaults to ``\"en\"``.
        metadata: Arbitrary key-value store for stage-specific data
            (chunk boundaries, quality scores, provenance hashes, etc.).

    Usage::

        from polygraph._shared import Document

        doc = Document(
            content=\"Albert Einstein was a theoretical physicist...\",
            source=\"einstein.txt\",
            doc_id=\"einstein_001\",
            language=\"en\",
        )
    """

    content: str
    source: str = ""
    doc_id: str = ""
    language: str = "en"
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"Document(id={self.doc_id!r}, source={self.source!r}, "
            f"lang={self.language!r}, content={preview!r}...)"
        )


@dataclass
class PreprocessResult:
    """Container returned by preprocessing — chunks plus optional extracted data.

    When an ``"extract"`` stage is included in the preprocessing pipeline,
    entities and triples are accumulated here and ``build_kg`` can skip
    re-extraction.  This allows extraction to happen at any point in the
    preprocessing sequence (e.g., before chunking).

    Attributes:
        chunks: Clean, deduplicated Document chunks.
        entities: Entities extracted during preprocessing (if any).
        triples: Relation triples extracted during preprocessing (if any).
        raw_docs: Raw documents before chunking, for doc-relation extraction.
    """

    chunks: list[Document] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[tuple] = field(default_factory=list)
    raw_docs: list[Document] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary payload from custom preprocessing stages (embeddings,
    topic labels, clusters, etc.).  Downstream stages access it via
    ``result.extra[\"my_key\"]``."""
