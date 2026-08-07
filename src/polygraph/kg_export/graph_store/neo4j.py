"""Neo4j-backed graph store — live, queryable access via Cypher."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from polygraph.kg_export.graph_store._base import GraphStore


class Neo4jGraphStore(GraphStore):
    """``GraphStore`` backed by a live Neo4j database.

    All reads are executed as Cypher queries, so no part of the graph is
    materialised in Python RAM.  Connection parameters default to the
    ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` environment variables
    and can be overridden in the constructor.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        from polygraph.kg_export.neo4j.upload import _get_connection

        # Retained so a pipeline can be pointed at this store and reuse the
        # same connection parameters (no need to repeat credentials).
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = _get_connection(uri=uri, user=user, password=password)
        self._closed = False

    # ── Internals ───────────────────────────────────────────────

    def _session(self) -> Any:
        return self._driver.session()

    def _single(self, query: str, **params: Any) -> Any:
        """Run a query returning a single value (or ``None``)."""
        with self._session() as session:
            record = session.run(query, **params).single()
            return record[0] if record else None

    # ── GraphStore ──────────────────────────────────────────────

    def number_of_nodes(self) -> int:
        return self._single("MATCH (n) RETURN count(n)")

    def number_of_edges(self) -> int:
        return self._single("MATCH ()-[r]->() RETURN count(r)")

    def nodes(self, data: bool = False) -> Iterable[Any]:
        with self._session() as session:
            for record in session.run("MATCH (n) RETURN properties(n) AS props"):
                props = dict(record["props"])
                node_id = str(props.get("id", ""))
                yield (node_id, props) if data else node_id

    def edges(self, data: bool = False) -> Iterable[Any]:
        with self._session() as session:
            for record in session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN a.id AS source, b.id AS target,
                       properties(r) AS props, type(r) AS rel_type
                """
            ):
                source = str(record["source"])
                target = str(record["target"])
                edge = dict(record["props"])
                edge.setdefault("type", record["rel_type"])
                yield (source, target, edge) if data else (source, target)

    def has_node(self, node: str) -> bool:
        return bool(self._single("MATCH (n {id: $id}) RETURN count(n)", id=node))

    def get_node(self, node: str) -> dict[str, Any] | None:
        with self._session() as session:
            record = session.run(
                "MATCH (n {id: $id}) RETURN properties(n) AS props", id=node
            ).single()
        if record is None:
            return None
        return dict(record["props"])

    def has_edge(self, u: str, v: str) -> bool:
        return bool(self._single("MATCH (a {id: $u})-[r]->(b {id: $v}) RETURN count(r)", u=u, v=v))

    def get_edge(self, u: str, v: str) -> dict[str, Any] | None:
        with self._session() as session:
            record = session.run(
                """
                MATCH (a {id: $u})-[r]->(b {id: $v})
                RETURN properties(r) AS props, type(r) AS rel_type
                """,
                u=u,
                v=v,
            ).single()
        if record is None:
            return None
        edge = dict(record["props"])
        edge.setdefault("type", record["rel_type"])
        return edge

    def successors(self, node: str) -> Iterable[str]:
        with self._session() as session:
            for record in session.run("MATCH (a {id: $id})-[r]->(b) RETURN b.id AS id", id=node):
                yield str(record["id"])

    def predecessors(self, node: str) -> Iterable[str]:
        with self._session() as session:
            for record in session.run("MATCH (a)-[r]->(b {id: $id}) RETURN a.id AS id", id=node):
                yield str(record["id"])

    def in_degree(self, node: str) -> int:
        return self._single("MATCH ()-[r]->(n {id: $id}) RETURN count(r)", id=node)

    def out_degree(self, node: str) -> int:
        return self._single("MATCH (n {id: $id})-[r]->() RETURN count(r)", id=node)

    def search(self, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._session() as session:
            records = session.run(
                """
                MATCH (n)
                WHERE n.id CONTAINS $q OR n.name CONTAINS $q
                RETURN properties(n) AS props
                LIMIT $limit
                """,
                q=query,
                limit=limit,
            )
            return [dict(record["props"]) for record in records]

    def close(self) -> None:
        if not self._closed:
            self._driver.close()
            self._closed = True


__all__ = ["Neo4jGraphStore"]
