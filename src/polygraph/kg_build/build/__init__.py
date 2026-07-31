"""Selectable graph-construction methods."""

from polygraph._shared import Ontology
from polygraph.kg_build.build.networkx import GraphBuilder
from polygraph.kg_build.build.registry import BUILD_METHODS, create_build_method
from polygraph.kg_build.build.sqlite import SQLiteGraphBuilder


def from_resolved(
    resolved,
    triples,
    *,
    method: str = "networkx",
    ontology: Ontology | None = None,
    **kwargs,
):
    """Build a graph with the selected method.

    Args:
        resolved: List of resolved entity dicts.
        triples: List of (subject, predicate, object, ...) tuples.
        method: ``"networkx"`` (in-memory) or ``"sqlite"`` (file-backed).
        ontology: Optional ontology for validation.
        **kwargs: Forwarded to the builder constructor (e.g. ``db_path`` for SQLite).

    Returns:
        An ``nx.DiGraph`` or ``SQLiteGraph`` (both support the same API for
        evaluation and export).
    """
    builder = create_build_method(method, ontology=ontology, **kwargs)
    return builder.build(resolved, triples)


__all__ = [
    "BUILD_METHODS",
    "GraphBuilder",
    "SQLiteGraphBuilder",
    "create_build_method",
    "from_resolved",
]
