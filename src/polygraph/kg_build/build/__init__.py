"""Storage-agnostic graph construction — one routine, any backend.

The same ``build_kg_into`` routine writes through any ``GraphWriter``:
``NetworkXGraphWriter`` (in-memory), ``SQLiteGraphWriter`` (file-backed), or
``Neo4jGraphBuilder`` (streamed). Swap the writer, keep the pipeline code.
"""

from polygraph.kg_build.build.networkx import GraphBuilder
from polygraph.kg_build.build.sqlite import SQLiteGraph, SQLiteGraphWriter
from polygraph.kg_build.build.writer import (
    GraphWriter,
    NetworkXGraphWriter,
    build_kg_into,
)

__all__ = [
    "GraphBuilder",
    "GraphWriter",
    "NetworkXGraphWriter",
    "SQLiteGraph",
    "SQLiteGraphWriter",
    "build_kg_into",
]
