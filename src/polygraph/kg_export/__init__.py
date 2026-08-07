"""KG Export: save graphs to files or graph databases.

Function API:
    from polygraph.kg_export import exporter

    # File formats
    exporter.to_json(graph, entities, triples, "output/knowledge_graph.json")
    exporter.to_graphml(graph, "output/knowledge_graph.graphml")

    # Graph databases (Neo4j is the only currently supported backend)
    exporter.to_graph_db("output/knowledge_graph.json", backend="neo4j", clear=False)

    # Backward-compatible alias
    exporter.to_neo4j("output/knowledge_graph.json", clear=False)

    # Storage-agnostic graph interaction (read side)
    store = create_graph_store("neo4j")  # or networkx / sqlite / json / graphml
    store.number_of_nodes()

Class API:
    from polygraph.kg_export import GraphExporter, GraphStore, create_graph_store
"""

from types import SimpleNamespace

from polygraph.kg_export.graph_store import GraphStore, create_graph_store
from polygraph.kg_export.json.exporter import GraphExporter

# ── Function API ────────────────────────────────────────────────

exporter = SimpleNamespace()


def _to_json(graph, entities, triples, path, metadata=None):
    GraphExporter().export(graph, entities, triples, output_dir=path.parent, formats=["json"])


def _to_graphml(graph, path):
    GraphExporter().export(graph, [], [], output_dir=path.parent, formats=["graphml"])


def _to_graph_db(json_path, *, backend: str = "neo4j", clear: bool = False, **kwargs):
    """Upload a knowledge graph JSON file to a graph database.

    Args:
        json_path: Path to ``knowledge_graph.json``.
        backend: Which graph DB to use. Currently only ``"neo4j"`` is supported.
        clear: If True, wipe the database before uploading.
        **kwargs: Forwarded to the backend constructor (e.g. ``uri``, ``user``,
            ``password`` for Neo4j).

    Example::

        exporter.to_graph_db("output/kg.json", backend="neo4j", clear=True)
        exporter.to_graph_db("output/kg.json", backend="neo4j",
                             uri="bolt://localhost:7687", user="neo4j", password="secret")
    """
    from polygraph.kg_export.graph_db import BACKENDS

    cls = BACKENDS.get(backend)
    if cls is None:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown graph DB backend: '{backend}'. Available: {available}")
    uploader = cls(**kwargs)
    uploader.upload(json_path, clear=clear)


def _to_neo4j(json_path, clear=False):
    """Upload to Neo4j. Backward-compatible alias for ``to_graph_db(backend="neo4j")``."""
    _to_graph_db(json_path, backend="neo4j", clear=clear)


exporter.to_json = _to_json
exporter.to_graphml = _to_graphml
exporter.to_graph_db = _to_graph_db
exporter.to_neo4j = _to_neo4j

__all__ = ["GraphExporter", "GraphStore", "create_graph_store", "exporter"]
