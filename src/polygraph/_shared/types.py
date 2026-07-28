"""Shared types used across all pipeline modules."""

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single document with metadata — the universal currency between stages."""

    content: str
    source: str = ""
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Document(id={self.doc_id!r}, source={self.source!r}, content={preview!r}...)"
