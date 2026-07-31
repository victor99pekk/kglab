"""Shared foundation — zero internal dependencies, everything depends on this."""

from polygraph._shared.config import (
    GraphBackend,
    Language,
    Ontology,
    PipelineConfig,
    load_config,
)
from polygraph._shared.identity import chunk_id, document_id, entity_id, stable_id
from polygraph._shared.language import (
    ALL_LANGUAGES,
    SkippedDocs,
    filter_by_language,
    get_language_support,
)
from polygraph._shared.language_discovery import (
    discover_pipeline_languages,
    discover_stage_languages,
)
from polygraph._shared.stage_config import (
    BuildConfig,
    DocumentRelationConfig,
    EvalConfig,
    ExportConfig,
    ExtractionConfig,
    PreprocessConfig,
    PreprocessStage,
    ResolutionConfig,
)
from polygraph._shared.types import Document

__all__ = [
    "ALL_LANGUAGES",
    "BuildConfig",
    "Document",
    "DocumentRelationConfig",
    "EvalConfig",
    "ExportConfig",
    "ExtractionConfig",
    "GraphBackend",
    "Language",
    "Ontology",
    "PipelineConfig",
    "PreprocessConfig",
    "PreprocessStage",
    "ResolutionConfig",
    "SkippedDocs",
    "chunk_id",
    "discover_pipeline_languages",
    "discover_stage_languages",
    "document_id",
    "entity_id",
    "filter_by_language",
    "get_language_support",
    "load_config",
    "stable_id",
]
