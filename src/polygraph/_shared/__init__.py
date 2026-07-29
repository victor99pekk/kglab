"""Shared foundation — zero internal dependencies, everything depends on this."""

from polygraph._shared.config import (
    GraphBackend,
    Language,
    Ontology,
    PipelineConfig,
    load_config,
)
from polygraph._shared.identity import chunk_id, document_id, entity_id, stable_id
from polygraph._shared.types import Document

__all__ = [
    "Document",
    "GraphBackend",
    "Language",
    "Ontology",
    "PipelineConfig",
    "chunk_id",
    "document_id",
    "entity_id",
    "load_config",
    "stable_id",
]
