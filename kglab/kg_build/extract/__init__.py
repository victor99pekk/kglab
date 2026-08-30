"""Selectable entity, relation, and joint extraction methods."""

from collections.abc import Iterable
from typing import Any

from kglab._shared import Ontology
from kglab.kg_build.extract._base import (
    Entity,
    EntityExtractor,
    JointExtractor,
    RelationExtractorMethod,
)
from kglab.kg_build.extract.entity import SimpleExtractor, SpacyExtractor
from kglab.kg_build.extract.joint import GraphGenExtractor
from kglab.kg_build.extract.registry import (
    ENTITY_METHODS,
    JOINT_METHODS,
    RELATION_METHODS,
    create_entity_method,
    create_joint_method,
    create_relation_method,
)
from kglab.kg_build.extract.relation import (
    CompositeRelationExtractor,
    OntologyRuleRelationExtractor,
    StructuredLLMRelationExtractor,
)


def _stamp_chunk_provenance(entities: list[Entity], chunk_id: str) -> None:
    """Record the chunk that produced every extracted entity."""
    if not chunk_id:
        return
    for entity in entities:
        if chunk_id not in entity.source_chunk_ids:
            entity.source_chunk_ids.append(chunk_id)


def with_methods(
    chunks: Iterable[Any],
    ontology: Ontology,
    *,
    entity_method: str = "spacy",
    relation_method: str = "ontology_rules",
    entity_options: dict[str, Any] | None = None,
    relation_options: dict[str, Any] | None = None,
):
    """Compose one entity method and one relation method."""
    entity_extractor = create_entity_method(entity_method, **(entity_options or {}))
    relation_extractor = create_relation_method(
        relation_method,
        ontology=ontology,
        **(relation_options or {}),
    )
    all_entities = []
    all_triples = []
    for chunk in chunks:
        entities = entity_extractor.extract(chunk.content)
        _stamp_chunk_provenance(entities, chunk.doc_id)
        triples = relation_extractor.extract(
            chunk.content,
            entities,
            source_chunk_id=chunk.doc_id,
        )
        all_entities.extend(entity.to_dict() for entity in entities)
        all_triples.extend(triples)
    return all_entities, all_triples


def jointly(
    chunks: Iterable[Any],
    ontology: Ontology,
    *,
    method: str = "graphgen",
    options: dict[str, Any] | None = None,
):
    """Run a method that jointly extracts entities and relations."""
    extractor = create_joint_method(method, ontology=ontology, **(options or {}))
    all_entities = []
    all_triples = []
    for chunk in chunks:
        entities, triples = extractor.extract(chunk.content, source_chunk_id=chunk.doc_id)
        _stamp_chunk_provenance(entities, chunk.doc_id)
        all_entities.extend(entity.to_dict() for entity in entities)
        all_triples.extend(triples)
    return all_entities, all_triples


def with_spacy(chunks, ontology: Ontology, model: str = "en_core_web_sm"):
    """Compatibility convenience function for the baseline composed method."""
    return with_methods(
        chunks,
        ontology,
        entity_method="spacy",
        relation_method="ontology_rules",
        entity_options={"model_name": model},
    )


def with_graphgen(
    chunks,
    ontology: Ontology,
    model: str = "deepseek-v4-pro",
    max_gleanings: int = 3,
):
    """Compatibility convenience function for GraphGen joint extraction."""
    return jointly(
        chunks,
        ontology,
        method="graphgen",
        options={"model_name": model, "max_gleanings": max_gleanings},
    )


__all__ = [
    "CompositeRelationExtractor",
    "ENTITY_METHODS",
    "Entity",
    "EntityExtractor",
    "GraphGenExtractor",
    "JOINT_METHODS",
    "JointExtractor",
    "OntologyRuleRelationExtractor",
    "RELATION_METHODS",
    "RelationExtractorMethod",
    "SimpleExtractor",
    "SpacyExtractor",
    "StructuredLLMRelationExtractor",
    "create_entity_method",
    "create_joint_method",
    "create_relation_method",
    "jointly",
    "with_graphgen",
    "with_methods",
    "with_spacy",
]
