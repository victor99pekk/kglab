"""Selectable graph-construction methods."""

from polygraph._shared import GraphBackend, Ontology
from polygraph.kg_build.build.networkx import GraphBuilder
from polygraph.kg_build.build.registry import BUILD_METHODS, create_build_method


def from_resolved(
    resolved,
    triples,
    *,
    method: str = "networkx",
    ontology: Ontology | None = None,
):
    """Build a graph with the selected method."""
    backend = GraphBackend.NETWORKX if method == "networkx" else method
    builder = create_build_method(method, ontology=ontology, backend=backend)
    return builder.build(resolved, triples)


__all__ = ["BUILD_METHODS", "GraphBuilder", "create_build_method", "from_resolved"]
