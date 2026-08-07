"""KG generation: extract entities, resolve duplicates, then build a graph.

Each pipeline stage is a package. Each selectable method has its own module.
"""

from polygraph.kg_build import build, extract, resolve
from polygraph.kg_build.build import (
    GraphBuilder,
    GraphWriter,
    NetworkXGraphWriter,
    SQLiteGraphBuilder,
    build_kg_into,
)
from polygraph.kg_build.build.sqlite import SQLiteGraph
from polygraph.kg_build.extract import (
    Entity,
    EntityExtractor,
    GraphGenExtractor,
    OntologyRuleRelationExtractor,
    SimpleExtractor,
    SpacyExtractor,
    StructuredLLMRelationExtractor,
)
from polygraph.kg_build.resolve import EntityResolver

__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityResolver",
    "GraphBuilder",
    "GraphGenExtractor",
    "GraphWriter",
    "NetworkXGraphWriter",
    "OntologyRuleRelationExtractor",
    "SimpleExtractor",
    "SpacyExtractor",
    "SQLiteGraph",
    "SQLiteGraphBuilder",
    "StructuredLLMRelationExtractor",
    "build",
    "build_kg_into",
    "extract",
    "resolve",
]
