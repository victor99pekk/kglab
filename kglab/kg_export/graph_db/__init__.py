"""Graph database upload — abstract interface and implementations.

Schema::

    Uploader             (ABC — graph → graph DB)
    ├── Neo4jUploader    (the only currently supported backend)

Usage::

    from kglab.kg_export.graph_db import Neo4jUploader

    uploader = Neo4jUploader(uri="bolt://localhost:7687", user="neo4j", password="...")
    uploader.upload("output/knowledge_graph.json", clear=True)

Or through the convenience API::

    from kglab.kg_export import exporter
    exporter.to_graph_db("output/kg.json", backend="neo4j", clear=False)
"""

from kglab.kg_export.graph_db._base import GraphDBUploader
from kglab.kg_export.graph_db.neo4j import Neo4jUploader

#: Registry mapping backend name strings → GraphDBUploader subclasses.
BACKENDS: dict[str, type[GraphDBUploader]] = {
    "neo4j": Neo4jUploader,
}

__all__ = ["BACKENDS", "GraphDBUploader", "Neo4jUploader"]
