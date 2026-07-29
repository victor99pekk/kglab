"""KG generation: extract entities, resolve duplicates, then build a graph.

Each pipeline stage is a package. Each selectable method has its own module.
"""

from polygraph.kg_build import build, extract, resolve
from polygraph.kg_build.build import GraphBuilder
from polygraph.kg_build.extract import (
    EnglishExtractor,
    Entity,
    EntityExtractor,
    GraphGenExtractor,
    OntologyRuleRelationExtractor,
    SimpleExtractor,
    StructuredLLMRelationExtractor,
)
from polygraph.kg_build.resolve import EntityResolver

__all__ = [
    "EnglishExtractor",
    "Entity",
    "EntityExtractor",
    "EntityResolver",
    "GraphBuilder",
    "GraphGenExtractor",
    "OntologyRuleRelationExtractor",
    "SimpleExtractor",
    "StructuredLLMRelationExtractor",
    "build",
    "extract",
    "resolve",
]
