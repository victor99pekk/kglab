"""SQLite-backed graph store."""

from __future__ import annotations

from typing import Any

from polygraph.kg_build.build.sqlite import SQLiteGraph
from polygraph.kg_export.graph_store._base import GraphStore


class SQLiteGraphStore(GraphStore):
    """``GraphStore`` backed by a file-backed ``SQLiteGraph``."""

    def __init__(self, graph: SQLiteGraph) -> None:
        self._graph = graph

    @property
    def graph(self) -> SQLiteGraph:
        """The underlying ``SQLiteGraph`` object."""
        return self._graph

    # ── GraphStore ──────────────────────────────────────────────

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def nodes(self, data: bool = False):
        return self._graph._iter_nodes(data=data)

    def edges(self, data: bool = False):
        return self._graph._iter_edges(data=data)

    def has_node(self, node: str) -> bool:
        return node in self._graph

    def get_node(self, node: str) -> dict[str, Any] | None:
        if node not in self._graph:
            return None
        return dict(self._graph.nodes[node])

    def has_edge(self, u: str, v: str) -> bool:
        return self._graph.has_edge(u, v)

    def get_edge(self, u: str, v: str) -> dict[str, Any] | None:
        if not self._graph.has_edge(u, v):
            return None
        return dict(self._graph.edges[u, v])

    def successors(self, node: str):
        return self._graph.successors(node)

    def predecessors(self, node: str):
        return (u for u, v in self._graph._iter_edges(data=False) if v == node)

    def in_degree(self, node: str) -> int:
        # Direct SQL lookup (SQLiteGraph exposes its connection internally).
        row = self._graph._conn.execute(
            "SELECT COUNT(*) AS c FROM edges WHERE object = ?", (node,)
        ).fetchone()
        return row["c"] if row else 0

    def out_degree(self, node: str) -> int:
        row = self._graph._conn.execute(
            "SELECT COUNT(*) AS c FROM edges WHERE subject = ?", (node,)
        ).fetchone()
        return row["c"] if row else 0

    def close(self) -> None:
        self._graph.close()


__all__ = ["SQLiteGraphStore"]
