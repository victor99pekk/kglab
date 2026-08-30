"""Shared foundation — zero internal dependencies, everything depends on this."""

from kglab._shared.config import (
    GraphBackend,
    Language,
    Ontology,
)
from kglab._shared.identity import chunk_id, document_id, entity_id, stable_id
from kglab._shared.language import (
    ALL_LANGUAGES,
    SkippedDocs,
    filter_by_language,
    get_language_support,
)
from kglab._shared.language_discovery import (
    discover_pipeline_languages,
    discover_stage_languages,
)
from kglab._shared.stage_config import (
    BuildConfig,
    DocumentRelationConfig,
    EvalConfig,
    ExportConfig,
    ExtractionConfig,
    LinkingConfig,
    ResolutionConfig,
)
from kglab._shared.types import Document, PreprocessResult

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
    "LinkingConfig",
    "Ontology",
    "PreprocessResult",
    "ResolutionConfig",
    "SkippedDocs",
    "chunk_id",
    "discover_pipeline_languages",
    "discover_stage_languages",
    "document_id",
    "entity_id",
    "filter_by_language",
    "get_language_support",
    "stable_id",
]
