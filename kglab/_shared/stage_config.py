"""Typed configuration for pipeline stages — code-is-truth, YAML conforms to this.

These dataclasses replace raw dict-digging in pipeline ``build_kg()`` methods.
The YAML ``pipeline`` block is validated and parsed into these objects so
pipelines receive fully typed, defaulted configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_DEFAULT_DOCUMENT_RELATION_METHODS = ("hyperlink",)


@dataclass
class DocumentRelationConfig:
    """Document-to-document relation extraction settings."""

    methods: list[str] = field(default_factory=lambda: list(_DEFAULT_DOCUMENT_RELATION_METHODS))
    method_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DocumentRelationConfig:
        if not data:
            return cls()

        raw_methods = data.get("methods")
        if raw_methods is None:
            methods = list(_DEFAULT_DOCUMENT_RELATION_METHODS)
        elif not isinstance(raw_methods, list) or not all(
            isinstance(method, str) and method for method in raw_methods
        ):
            raise ValueError("document_relation.methods must be a list of non-empty strings")
        else:
            methods = list(raw_methods)

        enabled = data.get("enabled", True)
        if enabled and not methods:
            raise ValueError(
                "document_relation.methods must contain at least one method when enabled"
            )

        return cls(
            methods=methods,
            method_options=data.get("method_options", {}),
            enabled=enabled,
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


# ── Evaluation ─────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Evaluation configuration (recorded in run manifests).

    Controls which evaluation passes run after KG construction.
    Evaluation itself is external to pipelines — see
    ``kglab.kg_eval.evaluate_kg``.
    """

    quality_enabled: bool = True
    structural_enabled: bool = True
    accuracy_enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvalConfig:
        if not data:
            return cls()
        return cls(
            quality_enabled=data.get("quality_enabled", True),
            structural_enabled=data.get("structural_enabled", True),
            accuracy_enabled=data.get("accuracy_enabled", False),
        )


# ── Export ─────────────────────────────────────────────────────


@dataclass
class ExportConfig:
    """Export stage configuration.

    Controls which formats the KG is written to after construction.
    """

    formats: list[str] = field(default_factory=lambda: ["json"])
    """Export format names: ``"json"``, ``"graphml"``, ``"neo4j"``."""

    neo4j_clear: bool = False
    """If ``True``, wipe the Neo4j database before uploading."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExportConfig:
        if not data:
            return cls()
        return cls(
            formats=data.get("formats", ["json"]),
            neo4j_clear=data.get("neo4j_clear", False),
        )


# ── Entity Linking ─────────────────────────────────────────────


@dataclass
class LinkingConfig:
    """Entity linking stage configuration.

    Links resolved entities to entries in an external knowledge base
    (Wikidata, DBpedia, custom taxonomy, etc.).
    """

    method: str = ""
    """Registered linker name (e.g. ``"wikidata"``). Required when enabled."""

    enabled: bool = False
    """Set ``True`` to run linking during the pipeline."""

    options: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the linker constructor."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LinkingConfig:
        if not data:
            return cls()
        return cls(
            method=data.get("method", ""),
            enabled=data.get("enabled", False),
            options=data.get("options", {}),
        )
