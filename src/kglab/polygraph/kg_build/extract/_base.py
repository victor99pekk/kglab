"""Shared extraction contracts and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeAlias

from kglab._shared import entity_id

Triple: TypeAlias = tuple[str, ...]


@dataclass
class Entity:
    """A single extracted entity with GraphRAG-ready properties."""

    name: str
    label: str
    mentions: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    description: str = ""
    source: str = ""
    embedding: list[float] | None = None
    node_id: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.node_id or entity_id(self.label, self.name)

    @property
    def aliases(self) -> list[str]:
        return sorted({mention.casefold() for mention in self.mentions})

    @property
    def display_name(self) -> str:
        """Human-readable display name (defaults to ``name``)."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.label,
            "aliases": list(self.aliases),
            "description": self.description,
            "confidenceScore": self.confidence,
            "importanceScore": 0.0,
            "source": [self.source] if self.source else [],
            "source_chunk_ids": list(dict.fromkeys(self.source_chunk_ids)),
            "embedding": self.embedding,
            "updatedAt": datetime.now(UTC).isoformat(),
        }


class EntityExtractor(ABC):
    """Contract implemented by every entity extraction method."""

    @abstractmethod
    def extract(self, text: str) -> list[Entity]: ...


class RelationExtractorMethod(ABC):
    """Contract implemented by every relation extraction method."""

    @abstractmethod
    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[Triple]: ...


class JointExtractor(ABC):
    """Contract for methods that jointly extract entities and relations."""

    @abstractmethod
    def extract(
        self,
        text: str,
        source_chunk_id: str = "",
    ) -> tuple[list[Entity], list[Triple]]: ...
