"""Knowledge Graph generation: extract entities → resolve duplicates → build graph.

Public API:
    from polygraph.kg_build import (
        extract_entities, resolve_entities, build_graph,
        EnglishExtractor, GraphGenExtractor, RelationExtractor,
        EntityResolver, GraphBuilder,
    )
"""

from polygraph.kg_build.build import GraphBuilder
from polygraph.kg_build.extract.entities import (
    EnglishExtractor,
    Entity,
    EntityExtractor,
    SimpleExtractor,
)
from polygraph.kg_build.extract.graphgen import GraphGenExtractor
from polygraph.kg_build.extract.relations import RelationExtractor
from polygraph.kg_build.resolve import EntityResolver

__all__ = [
    "EnglishExtractor",
    "Entity",
    "EntityExtractor",
    "EntityResolver",
    "GraphBuilder",
    "GraphGenExtractor",
    "RelationExtractor",
    "SimpleExtractor",
]
