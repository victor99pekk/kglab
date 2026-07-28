"""KG Export: save graphs to files or Neo4j.

Function API:
    from polygraph.kg_export import exporter

    exporter.to_json(graph, entities, triples, "output/knowledge_graph.json")
    exporter.to_graphml(graph, "output/knowledge_graph.graphml")

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


exporter.to_json = _to_json
exporter.to_graphml = _to_graphml

__all__ = ["GraphExporter", "exporter"]
