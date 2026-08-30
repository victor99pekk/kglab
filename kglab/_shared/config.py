"""Knowledge Graph Generator — core configuration and ontology definitions."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class Language(StrEnum):
    ENGLISH = "en"


class GraphBackend(StrEnum):
    NETWORKX = "networkx"
    NEO4J = "neo4j"


@dataclass
class Ontology:
    """Defines the schema for a knowledge graph: entity types, relations, attributes.

    Load from a YAML file via ``Ontology.from_yaml(path)``.  The YAML must
    contain top-level keys ``entity_types``, ``relationship_types``, and
    optionally ``attributes``.

    Entity types map a label (e.g. ``PERSON``) to a dict with at least a
    ``description`` key.

    Relationship types map a predicate (e.g. ``works_at``) to a dict with
    optional ``domain``, ``range``, and ``symmetric`` keys.
    """

    entity_types: dict[str, dict[str, str]] = field(default_factory=dict)
    relationship_types: dict[str, dict[str, Any]] = field(default_factory=dict)
    attributes: dict[str, list[str]] = field(default_factory=dict)

    # ── helpers ─────────────────────────────────────────────────

    def get_entity_type_names(self) -> list[str]:
        """Return the canonical (upper-case) entity type labels."""
        return list(self.entity_types.keys())

    def get_relation_patterns(self) -> list[tuple[str, str, str, bool]]:
        """Return (domain, range, predicate, symmetric) patterns for rule-based extraction.

        Only semantic relations with both a *domain* and a *range* are
        extractable. Structural relations are created explicitly by the
        pipeline, while untyped relations are metadata rather than an
        instruction to connect every co-occurring entity pair.
        """
        patterns: list[tuple[str, str, str, bool]] = []
        for predicate, info in self.relationship_types.items():
            if info.get("kind") == "structural":
                continue

            domain = str(info.get("domain", "")).strip()
            range_ = str(info.get("range", "")).strip()
            symmetric = bool(info.get("symmetric", False))

            if domain and range_:
                patterns.append((domain, range_, predicate, symmetric))
        return patterns

    def get_structural_relation_types(self) -> dict[str, tuple[str, str]]:
        """Return structural predicates and their required endpoint types."""
        structural: dict[str, tuple[str, str]] = {}
        for predicate, info in self.relationship_types.items():
            if info.get("kind") != "structural":
                continue
            domain = str(info.get("domain", "")).strip()
            range_ = str(info.get("range", "")).strip()
            if not domain or not range_:
                raise ValueError(f"Structural relation '{predicate}' must define domain and range")
            structural[predicate] = (domain, range_)
        return structural

    @classmethod
    def from_yaml(cls, path: Path) -> "Ontology":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            entity_types=data.get("entity_types", {}),
            relationship_types=data.get("relationship_types", {}),
            attributes=data.get("attributes", {}),
        )
