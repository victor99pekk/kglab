"""Typed configuration for pipeline stages — code-is-truth, YAML conforms to this.

These dataclasses replace raw dict-digging in pipeline ``build_kg()`` methods.
The YAML ``pipeline`` block is validated and parsed into these objects so
pipelines receive fully typed, defaulted configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentRelationConfig:
    """Document-to-document relation extraction settings."""

    methods: list[str] = field(
        default_factory=lambda: [
            "hyperlink",
            "shared_authors",
            "series",
            "citation",
            "shared_references",
        ]
    )
    method_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DocumentRelationConfig:
        if not data:
            return cls()
        return cls(
            methods=data.get("methods", cls.methods),
            method_options=data.get("method_options", {}),
            enabled=data.get("enabled", True),
        )


@dataclass
class ExtractionConfig:
    """Entity and relation extraction stage configuration."""

    mode: str = "composed"  # composed | joint
    entity_method: str = "spacy"
    relation_method: str = "ontology_rules"
    joint_method: str = "graphgen"
    entity_options: dict[str, Any] | None = None
    relation_options: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    document_relation: DocumentRelationConfig = field(default_factory=DocumentRelationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExtractionConfig:
        if not data:
            return cls()
        legacy_method = data.get("method")
        doc_rel = DocumentRelationConfig.from_dict(data.get("document_relation"))
        return cls(
            mode=data.get(
                "mode",
                "joint" if legacy_method == "graphgen" else "composed",
            ),
            entity_method=data.get("entity_method", "spacy"),
            relation_method=data.get("relation_method", "ontology_rules"),
            joint_method=data.get("joint_method", "graphgen"),
            entity_options=data.get("entity_options"),
            relation_options=data.get("relation_options", {}),
            options=data.get("options", {}),
            document_relation=doc_rel,
        )


@dataclass
class ResolutionConfig:
    """Entity resolution stage configuration."""

    method: str = "string"
    threshold: float = 0.85
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ResolutionConfig:
        if not data:
            return cls()
        return cls(
            method=data.get("method", "string"),
            threshold=data.get("threshold", 0.85),
            options=data.get("options", {}),
        )


@dataclass
class BuildConfig:
    """Graph construction stage configuration."""

    method: str = "networkx"
    graphml: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BuildConfig:
        if not data:
            return cls()
        return cls(
            method=data.get("method", "networkx"),
            graphml=data.get("graphml", False),
        )
