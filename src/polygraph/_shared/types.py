"""Shared types used across all pipeline modules."""

from dataclasses import dataclass, field


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
