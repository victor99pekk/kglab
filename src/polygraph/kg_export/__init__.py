"""KG Export: save graphs to files or Neo4j.

Function API:
    from polygraph.kg_export import exporter

    exporter.to_json(graph, entities, triples, "output/knowledge_graph.json")
    exporter.to_graphml(graph, "output/knowledge_graph.graphml")
    exporter.to_neo4j("output/knowledge_graph.json", clear=False)

Class API:
    from polygraph.kg_export import GraphExporter
"""

from types import SimpleNamespace

from polygraph.kg_export.exporter import GraphExporter

# ── Function API ────────────────────────────────────────────────

exporter = SimpleNamespace()


def _to_json(graph, entities, triples, path, metadata=None):
    GraphExporter().export(graph, entities, triples, output_dir=path.parent, formats=["json"])


def _to_graphml(graph, path):
    GraphExporter().export(graph, [], [], output_dir=path.parent, formats=["graphml"])


def _to_neo4j(json_path, clear=False):
    """Upload a knowledge graph JSON file to Neo4j.

    Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD environment variables.
    Set clear=True to wipe the database first.
    """
    from polygraph.kg_export.neo4j.upload import upload_graph

    upload_graph(json_path, clear=clear)


exporter.to_json = _to_json
exporter.to_graphml = _to_graphml
exporter.to_neo4j = _to_neo4j

__all__ = ["GraphExporter", "exporter"]
