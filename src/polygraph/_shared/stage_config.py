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


# ── Preprocessing ──────────────────────────────────────────────


@dataclass
class PreprocessStage:
    """One named step in the preprocessing pipeline.

    Each stage maps to a registry under ``polygraph.preprocess``
    (e.g. ``"chunk"`` maps to ``polygraph.preprocess.chunk``).
    The ``method`` selects which function in that registry to call.
    """

    name: str
    """Stage name: ``"load"``, ``"clean"``, ``"link_normalize"``,
    ``"quality"``, ``"dedup"``, ``"chunk"``, ``"language_filter"``."""

    method: str = "default"
    """Method name within the stage registry."""

    enabled: bool = True
    """Set ``False`` to skip this stage."""

    options: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the stage function."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessStage:
        if not isinstance(data, dict):
            raise TypeError("PreprocessStage data must be a dict")
        name = data.get("name", "")
        if not name or not isinstance(name, str):
            raise ValueError("PreprocessStage requires a non-empty 'name'")
        return cls(
            name=name,
            method=data.get("method", "default"),
            enabled=data.get("enabled", True),
            options=data.get("options", {}),
        )


@dataclass
class PreprocessConfig:
    """Preprocessing pipeline configuration.

    Three tiers of customization, from simplest to most powerful:

    **Tier 0 — Zero config (sensible defaults):**

        Baseline(input_paths=["data/"], output_dir="output/")

    **Tier 1 — Simple knobs (covers 90% of use cases):**

        Baseline(
            input_paths=["data/"], output_dir="output/",
            preprocess=PreprocessConfig(
                chunk_method="semantic",
                chunk_target_tokens=300,
            ),
        )

    **Tier 2 — Full stage control (power users):**

        Baseline(
            input_paths=["data/"], output_dir="output/",
            preprocess=PreprocessConfig(stages=[
                PreprocessStage("load", "baseline"),
                PreprocessStage("chunk", "semantic", options={"target_tokens": 300}),
                PreprocessStage("dedup", "minhash"),
            ]),
        )

    When ``stages`` is provided (Tier 2), the simple knob fields are
    ignored.  When ``stages`` is ``None`` (Tier 1 / Tier 0), the
    simple knobs control the default stage sequence.
    """

    # ── Tier 1: Simple knobs ─────────────────────────────────

    clean_enabled: bool = True
    link_normalize_enabled: bool = True
    quality_min_chars: int = 50
    quality_min_words: int = 10
    doc_dedup_method: str = "minhash"
    doc_dedup_threshold: float = 0.85
    chunk_method: str = "sentence"
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 60
    chunk_semantic_threshold: float = 0.55
    chunk_semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_dedup_method: str = "minhash"
    chunk_dedup_threshold: float = 0.85

    # ── Tier 2: Explicit stage list ──────────────────────────

    stages: list[PreprocessStage] | None = None
    """When set, overrides all simple-knob fields above."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreprocessConfig:
        if not data:
            return cls()

        raw_stages = data.get("stages")
        if raw_stages is not None:
            if not isinstance(raw_stages, list):
                raise ValueError("preprocess.stages must be a list")
            stages = [PreprocessStage.from_dict(s) for s in raw_stages]
            return cls(stages=stages)

        return cls(
            clean_enabled=data.get("clean_enabled", True),
            link_normalize_enabled=data.get("link_normalize_enabled", True),
            quality_min_chars=data.get("quality_min_chars", 50),
            quality_min_words=data.get("quality_min_words", 10),
            doc_dedup_method=data.get("doc_dedup_method", "minhash"),
            doc_dedup_threshold=data.get("doc_dedup_threshold", 0.85),
            chunk_method=data.get("chunk_method", "sentence"),
            chunk_target_tokens=data.get("chunk_target_tokens", 450),
            chunk_overlap_tokens=data.get("chunk_overlap_tokens", 60),
            chunk_semantic_threshold=data.get("chunk_semantic_threshold", 0.55),
            chunk_semantic_model=data.get(
                "chunk_semantic_model", "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            chunk_dedup_method=data.get("chunk_dedup_method", "minhash"),
            chunk_dedup_threshold=data.get("chunk_dedup_threshold", 0.85),
        )


# ── Evaluation ─────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Evaluation stage configuration.

    Controls which evaluation passes run after KG construction.
    Accuracy evaluation requires an LLM client — pass one to
    ``Pipeline.evaluate(llm_client=...)``.
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
