"""In-memory (NetworkX-backed) graph store."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kglab.kg_export.graph_store._base import GraphStore


class NetworkXGraphStore(GraphStore):
    """``GraphStore`` backed by an in-memory ``nx.DiGraph``.

    Also accepts any object that mirrors the small ``nx.DiGraph`` read API
    (``number_of_nodes``, ``nodes(data=...)``, ``edges(data=...)``,
    ``successors``, ...) — e.g. ``SQLiteGraph``.
    """

    def __init__(
        self,
        graph: Any | None = None,
        *,
        loader: Callable[[], Any] | None = None,
    ) -> None:
        self._graph = graph
        self._loader = loader

    @property
    def graph(self) -> Any:
        """The underlying graph object (lazily loaded via ``loader``)."""
        if self._graph is None and self._loader is not None:
            self._graph = self._loader()
        return self._graph

    # ── GraphStore ──────────────────────────────────────────────

    def number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self.graph.number_of_edges()

    def nodes(self, data: bool = False):
        return self.graph.nodes(data=data)

    def edges(self, data: bool = False):
        return self.graph.edges(data=data)

    def has_node(self, node: str) -> bool:
        return node in self.graph

    def get_node(self, node: str) -> dict[str, Any] | None:
        if node not in self.graph:
            return None
        return dict(self.graph.nodes[node])

    def has_edge(self, u: str, v: str) -> bool:
        return self.graph.has_edge(u, v)

    def get_edge(self, u: str, v: str) -> dict[str, Any] | None:
        if not self.graph.has_edge(u, v):
            return None
        return dict(self.graph.edges[u, v])

    def successors(self, node: str):
        return self.graph.successors(node)

    def predecessors(self, node: str):
        return self.graph.predecessors(node)

    def in_degree(self, node: str) -> int:
        return self.graph.in_degree(node)

    def out_degree(self, node: str) -> int:
        return self.graph.out_degree(node)

    def to_networkx(self) -> Any:
        return self.graph

    def close(self) -> None:
        """Nothing to release for an in-memory graph."""


__all__ = ["NetworkXGraphStore"]
