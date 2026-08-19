"""KG generation: extract entities, resolve duplicates, then build a graph.

Each pipeline stage is a package. Each selectable method has its own module.
"""

from kglab.kg_build import build, extract, resolve
from kglab.kg_build.build import (
    GraphBuilder,
    GraphWriter,
    NetworkXGraphWriter,
    SQLiteGraphWriter,
    build_kg_into,
)
from kglab.kg_build.build.sqlite import SQLiteGraph
from kglab.kg_build.extract import (
    Entity,
    EntityExtractor,
    GraphGenExtractor,
    OntologyRuleRelationExtractor,
    SimpleExtractor,
    SpacyExtractor,
    StructuredLLMRelationExtractor,
)
from kglab.kg_build.resolve import EntityResolver

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
    "SQLiteGraphWriter",
    "StructuredLLMRelationExtractor",
    "build",
    "build_kg_into",
    "extract",
    "resolve",
]
