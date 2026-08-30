"""Storage-agnostic graph read interface — query any graph backend uniformly.

Usage::

    from kglab.kg_export.graph_store import create_graph_store

    store = create_graph_store("neo4j")                       # env-configured
    store = create_graph_store("networkx", graph=g)           # in-memory
    store = create_graph_store("sqlite", graph=sqlite_graph)  # file-backed
    store = create_graph_store("json", path="output/knowledge_graph.json")
    store = create_graph_store("graphml", path="output/knowledge_graph.graphml")
"""

from __future__ import annotations

from typing import Any

from kglab.kg_export.graph_store._base import GraphStore
from kglab.kg_export.graph_store.json import GraphMLGraphStore, JSONGraphStore
from kglab.kg_export.graph_store.neo4j import Neo4jGraphStore
from kglab.kg_export.graph_store.networkx import NetworkXGraphStore
from kglab.kg_export.graph_store.sqlite import SQLiteGraphStore

#: Registry mapping backend name strings → GraphStore subclasses.
GRAPH_STORE_BACKENDS: dict[str, type[GraphStore]] = {
    "networkx": NetworkXGraphStore,
    "sqlite": SQLiteGraphStore,
    "neo4j": Neo4jGraphStore,
    "json": JSONGraphStore,
    "graphml": GraphMLGraphStore,
}


def create_graph_store(backend: str, **kwargs: Any) -> GraphStore:
    """Create a ``GraphStore`` for the named backend.

    Args:
        backend: ``"networkx"``, ``"sqlite"``, ``"neo4j"``, ``"json"``, or
            ``"graphml"``.
        **kwargs: Forwarded to the backend constructor. Common keys:
            ``graph`` (networkx / sqlite), ``path`` (json / graphml), and
            ``uri`` / ``user`` / ``password`` (neo4j).

    Raises:
        ValueError: If the backend name is unknown.
    """
    try:
        cls = GRAPH_STORE_BACKENDS[backend]
    except KeyError as exc:
        choices = ", ".join(sorted(GRAPH_STORE_BACKENDS))
        raise ValueError(f"Unknown graph store backend '{backend}'. Available: {choices}") from exc
    return cls(**kwargs)


__all__ = [
    "GRAPH_STORE_BACKENDS",
    "GraphMLGraphStore",
    "GraphStore",
    "JSONGraphStore",
    "Neo4jGraphStore",
    "NetworkXGraphStore",
    "SQLiteGraphStore",
    "create_graph_store",
]
