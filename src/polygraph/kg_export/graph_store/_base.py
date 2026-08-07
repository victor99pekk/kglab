"""Storage-agnostic graph read interface — interact with any backend uniformly.

The write side of graph construction is already abstracted by ``GraphWriter``
(``polygraph.kg_build.build.writer``).  ``GraphStore`` is its read-side
counterpart: pipelines, evaluators, and exporters interact with a graph
through this interface without knowing whether it lives in memory, in an
SQLite/JSON/GraphML file, or in a Neo4j database.

Usage::

    from polygraph.kg_export.graph_store import create_graph_store

    store = create_graph_store("neo4j")                    # env-configured
    store = create_graph_store("networkx", graph=g)        # in-memory
    store = create_graph_store("json", path="output/knowledge_graph.json")

    store.number_of_nodes()
    for node, data in store.nodes(data=True):
        ...
    for u, v, data in store.edges(data=True):
        ...
    matches = store.search("Einstein")
    store.close()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class GraphStore(ABC):
    """Read/interact with a knowledge graph without knowing how it is stored.

    Every storage backend (in-memory ``nx.DiGraph``, SQLite file, JSON or
    GraphML export, Neo4j database) implements this interface so that the
    same pipeline, evaluation, and export code works against any backend.

    Iterators are lazy by design — backends stream from their source instead
    of materialising the whole graph in RAM.  Use ``to_networkx()`` when you
    explicitly want an in-memory ``nx.DiGraph``.
    """

    # ── Counts ──────────────────────────────────────────────────

    @abstractmethod
    def number_of_nodes(self) -> int:
        """Number of nodes in the graph."""

    @abstractmethod
    def number_of_edges(self) -> int:
        """Number of directed edges in the graph."""

    # ── Iteration ───────────────────────────────────────────────

    @abstractmethod
    def nodes(self, data: bool = False) -> Iterable[Any]:
        """Yield node IDs, or ``(node_id, data_dict)`` pairs when ``data``."""

    @abstractmethod
    def edges(self, data: bool = False) -> Iterable[Any]:
        """Yield ``(u, v)`` pairs, or ``(u, v, data_dict)`` when ``data``."""

    # ── Lookups ─────────────────────────────────────────────────

    @abstractmethod
    def has_node(self, node: str) -> bool:
        """Return whether a node with the given ID exists."""

    @abstractmethod
    def get_node(self, node: str) -> dict[str, Any] | None:
        """Return the node's property dict, or ``None`` if absent."""

    @abstractmethod
    def has_edge(self, u: str, v: str) -> bool:
        """Return whether a directed edge ``u -> v`` exists."""

    @abstractmethod
    def get_edge(self, u: str, v: str) -> dict[str, Any] | None:
        """Return the edge's property dict, or ``None`` if absent."""

    # ── Traversal ───────────────────────────────────────────────

    @abstractmethod
    def successors(self, node: str) -> Iterable[str]:
        """Yield the IDs of nodes reachable via an outgoing edge."""

    @abstractmethod
    def predecessors(self, node: str) -> Iterable[str]:
        """Yield the IDs of nodes with an incoming edge to ``node``."""

    def neighbors(self, node: str) -> Iterable[str]:
        """Yield outgoing neighbors (NetworkX compatibility)."""
        return self.successors(node)

    @abstractmethod
    def in_degree(self, node: str) -> int:
        """Number of incoming edges."""

    @abstractmethod
    def out_degree(self, node: str) -> int:
        """Number of outgoing edges."""

    # ── Search & materialisation ────────────────────────────────

    def search(self, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return node data for nodes whose ID or ``name`` contains ``query``.

        Backends may override with a more efficient implementation (e.g.
        Cypher for Neo4j).
        """
        q = query.casefold()
        matches: list[dict[str, Any]] = []
        for node_id, data in self.nodes(data=True):
            name = str(data.get("name", ""))
            if q in str(node_id).casefold() or q in name.casefold():
                matches.append({**data, "id": node_id})
                if len(matches) >= limit:
                    break
        return matches

    def to_networkx(self) -> Any:
        """Materialise the graph as an in-memory ``nx.DiGraph``.

        Default implementation streams from ``nodes(data=True)`` and
        ``edges(data=True)``; backends may override for efficiency.
        """
        import networkx as nx

        graph = nx.DiGraph()
        for node, data in self.nodes(data=True):
            graph.add_node(node, **data)
        for u, v, data in self.edges(data=True):
            graph.add_edge(u, v, **data)
        return graph

    @property
    def stats(self) -> dict[str, int]:
        """Summary counts, compatible with the pipeline ``stats`` dict."""
        return {"num_nodes": self.number_of_nodes(), "num_edges": self.number_of_edges()}

    # ── Lifecycle ───────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None:
        """Release backend resources (connections, files). No-op for memory."""


__all__ = ["GraphStore"]
