"""SQLite graph writer — storage-agnostic build into a file-backed graph.

Implements the shared ``GraphWriter`` interface (see
``kglab.kg_build.build.writer``) so the same ``build_kg_into`` routine
used for in-memory (``NetworkXGraphWriter``) and Neo4j (``Neo4jGraphBuilder``)
builds also targets a SQLite database file. The returned ``SQLiteGraph``
object mimics ``nx.DiGraph`` enough for evaluation and export to work
transparently.

Usage::

    from kglab.kg_build.build import SQLiteGraphWriter, build_kg_into

    writer = SQLiteGraphWriter(db_path="output/knowledge_graph.db")
    build_kg_into(writer, chunks, entities, triples)
    graph = writer.graph  # → SQL queries, not RAM counts
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from kglab._shared import Ontology
from kglab.kg_build.build.writer import GraphWriter

# ── SQLite graph wrapper (drop-in for nx.DiGraph) ──────────────


class SQLiteGraph:
    """A file-backed graph that mirrors the ``nx.DiGraph`` API surface used by
    evaluators and exporters.

    Nodes and edges are stored in a ``.db`` file — only queried data enters RAM.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self.graph: dict[str, Any] = {}  # graph-level attrs (nx compatibility)
        self.nodes = _NodeView(self)
        self.edges = _EdgesView(self)

    # ── nx.DiGraph-compatible API ────────────────────────────

    def number_of_nodes(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM nodes").fetchone()
        return row["cnt"] if row else 0

    def number_of_edges(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM edges").fetchone()
        return row["cnt"] if row else 0

    def is_directed(self) -> bool:
        return True

    def is_multigraph(self) -> bool:
        return False

    def has_edge(self, u: str, v: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM edges WHERE subject = ? AND object = ? LIMIT 1", (u, v)
        ).fetchone()
        return row is not None

    def _iter_nodes(self, data: bool = False):
        rows = self._conn.execute("SELECT id, attrs FROM nodes").fetchall()
        for row in rows:
            node_id = row["id"]
            attrs = json.loads(row["attrs"]) if row["attrs"] else {}
            if data:
                yield node_id, attrs
            else:
                yield node_id

    def _iter_edges(self, data: bool = False):
        rows = self._conn.execute("SELECT subject, object, attrs FROM edges").fetchall()
        for row in rows:
            attrs = json.loads(row["attrs"]) if row["attrs"] else {}
            if data:
                yield row["subject"], row["object"], attrs
            else:
                yield row["subject"], row["object"]

    def __getitem__(self, key: str) -> dict[str, Any]:
        return self.nodes[key]

    def __iter__(self):
        """Iterate over node IDs (NetworkX compatibility)."""
        return self._iter_nodes(data=False)

    def neighbors(self, node: str):
        """Return an iterator over all neighbors (both in and out, for BFS)."""
        return self.successors(node)

    def __contains__(self, key: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM nodes WHERE id = ? LIMIT 1", (key,)).fetchone()
        return row is not None

    def successors(self, node: str):
        """Return an iterator over outgoing neighbors (for BFS traversal)."""
        rows = self._conn.execute("SELECT object FROM edges WHERE subject = ?", (node,)).fetchall()
        return (row["object"] for row in rows)

    def in_degree(self):
        """Return (node, in_degree) pairs."""
        rows = self._conn.execute(
            "SELECT object AS node, COUNT(*) AS deg FROM edges GROUP BY object"
        ).fetchall()
        return [(row["node"], row["deg"]) for row in rows]

    def out_degree(self):
        """Return (node, out_degree) pairs."""
        rows = self._conn.execute(
            "SELECT subject AS node, COUNT(*) AS deg FROM edges GROUP BY subject"
        ).fetchall()
        return [(row["node"], row["deg"]) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


class _NodeView:
    """Node view that supports both iteration (``.nodes()``) and subscript
    access (``.nodes[node_id]``), matching ``nx.DiGraph.nodes`` behavior."""

    def __init__(self, graph: SQLiteGraph) -> None:
        self._graph = graph

    def __call__(self, data: bool = False):
        """Iterate: ``graph.nodes()`` or ``graph.nodes(data=True)``."""
        return self._graph._iter_nodes(data=data)

    def __getitem__(self, key: str) -> dict[str, Any]:
        """Subscript access: ``graph.nodes[\"node_id\"]``."""
        row = self._graph._conn.execute("SELECT attrs FROM nodes WHERE id = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row["attrs"]) if row["attrs"] else {}

    def __iter__(self):
        return self._graph._iter_nodes(data=False)

    def __contains__(self, key: str) -> bool:
        row = self._graph._conn.execute(
            "SELECT 1 FROM nodes WHERE id = ? LIMIT 1", (key,)
        ).fetchone()
        return row is not None

    def __len__(self) -> int:
        return self._graph.number_of_nodes()


class _EdgesView:
    """Edge view supporting ``graph.edges(data=True)`` and ``graph.edges[u, v]``."""

    def __init__(self, graph: SQLiteGraph) -> None:
        self._graph = graph

    def __call__(self, data: bool = False):
        """Iterate: ``graph.edges()`` or ``graph.edges(data=True)``."""
        return self._graph._iter_edges(data=data)

    def __getitem__(self, key: tuple[str, str]) -> dict[str, Any]:
        """Subscript access: ``graph.edges[u, v]``."""
        u, v = key
        row = self._graph._conn.execute(
            "SELECT attrs FROM edges WHERE subject = ? AND object = ?", (u, v)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row["attrs"]) if row["attrs"] else {}

    def __iter__(self):
        return self._graph._iter_edges(data=False)

    def __contains__(self, key: tuple[str, str]) -> bool:
        u, v = key
        return self._graph.has_edge(u, v)


# ── Writer (storage-agnostic GraphWriter backend) ─────────────


class SQLiteGraphWriter(GraphWriter):
    """``GraphWriter`` that builds a file-backed knowledge graph in SQLite.

    Implements the shared ``GraphWriter`` interface so the same
    ``build_kg_into`` routine used by the in-memory (``NetworkXGraphWriter``)
    and Neo4j (``Neo4jGraphBuilder``) backends also targets a SQLite file —
    swap the writer to change where the graph is stored, keep the pipeline
    code.

    Node writes are buffered and flushed with ``executemany`` inside a single
    transaction (WAL mode); edges are merged read-modify-write, accumulating
    ``predicates`` / ``relations`` / ``source_texts`` / ``source_chunk_ids``
    arrays exactly like the batch builder did.

    Args:
        db_path: Where to write the ``.db`` file. Defaults to
            ``knowledge_graph.db``.
        ontology: Optional ontology for structural validation.
    """

    _NODE_FLUSH_THRESHOLD = 5000

    def __init__(
        self,
        db_path: str | Path | None = None,
        ontology: Ontology | None = None,
    ) -> None:
        self.db_path = str(db_path) if db_path else "knowledge_graph.db"
        self.ontology = ontology
        self._graph = SQLiteGraph(self.db_path)
        self._node_buffer: list[tuple[str, str]] = []
        self._node_count = 0
        self._edge_count = 0
        self._init_tables()

    def _init_tables(self) -> None:
        """Create the schema and start from a clean graph (per build)."""
        conn = self._graph._conn
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                attrs TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                subject TEXT,
                object TEXT,
                attrs TEXT,
                PRIMARY KEY (subject, object)
            )
            """
        )
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM edges")

    # ── GraphWriter implementation ───────────────────────────────

    def merge_document(
        self,
        doc_id: str,
        *,
        name: str = "",
        description: str = "",
        source: str = "",
        chunk_count: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self._buffer_node(
            {
                "id": doc_id,
                "type": "Document",
                "name": name or doc_id,
                "description": description,
                "source": source,
                "chunk_count": chunk_count,
                **(properties or {}),
            }
        )

    def merge_chunk(
        self,
        chunk_id: str,
        *,
        source: str = "",
        text: str = "",
        token_count: int = 0,
        index: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self._buffer_node(
            {
                "id": chunk_id,
                "type": "Chunk",
                "name": chunk_id,
                "source": source,
                "text": text,
                "tokenCount": token_count,
                "index": index,
                **(properties or {}),
            }
        )

    def merge_entity(
        self,
        entity_id: str,
        *,
        name: str = "",
        entity_type: str = "Entity",
        description: str = "",
        importance_score: float = 0.0,
        confidence_score: float = 1.0,
        embedding: list[float] | None = None,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self._buffer_node(
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "description": description,
                "importanceScore": importance_score,
                "confidenceScore": confidence_score,
                "embedding": embedding,
                "aliases": aliases or [],
                **(properties or {}),
            }
        )

    def merge_edge(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        *,
        evidence_sentence: str = "",
        source_chunk_id: str = "",
        description: str = "",
        weight: int = 1,
    ) -> None:
        self._merge_edge(
            source_id,
            target_id,
            predicate,
            evidence_sentence,
            source_chunk_id,
            description,
        )
        self._edge_count += 1

    def merge_structural_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> None:
        self._merge_edge(source_id, target_id, relationship_type, "", "", "")
        self._edge_count += 1

    def _merge_edge(
        self,
        subject: str,
        object_id: str,
        predicate: str,
        evidence_sentence: str,
        source_chunk_id: str,
        description: str,
    ) -> None:
        """Insert or merge a directed edge, accumulating predicates/evidence."""
        conn = self._graph._conn

        # Ensure both endpoints exist (minimal ENTITY nodes, like the batch
        # builder did — real nodes written via merge_* are left untouched).
        for nid in (subject, object_id):
            conn.execute(
                "INSERT OR IGNORE INTO nodes (id, attrs) VALUES (?, ?)",
                (nid, json.dumps({"id": nid, "type": "ENTITY", "name": nid})),
            )

        # Merge with an existing edge if present
        existing = conn.execute(
            "SELECT attrs FROM edges WHERE subject = ? AND object = ?",
            (subject, object_id),
        ).fetchone()

        if existing:
            edge_data = json.loads(existing["attrs"]) if existing["attrs"] else {}
            preds = edge_data.get("predicates", [])
            if predicate not in preds:
                preds.append(predicate)
            edge_data["predicates"] = preds
            edge_data["weight"] = len(preds)

            texts = edge_data.get("source_texts", [])
            if evidence_sentence and evidence_sentence not in texts:
                texts.append(evidence_sentence)
            edge_data["source_texts"] = texts

            chunks = edge_data.get("source_chunk_ids", [])
            if source_chunk_id and source_chunk_id not in chunks:
                chunks.append(source_chunk_id)
            edge_data["source_chunk_ids"] = chunks

            rels = edge_data.get("relations", [])
            rel_record = {
                "predicate": predicate,
                "evidence_sentence": evidence_sentence,
                "source_chunk_id": source_chunk_id,
            }
            if description:
                rel_record["description"] = description
            if rel_record not in rels:
                rels.append(rel_record)
            edge_data["relations"] = rels
            if description:
                edge_data["description"] = description

            conn.execute(
                "UPDATE edges SET attrs = ? WHERE subject = ? AND object = ?",
                (json.dumps(edge_data), subject, object_id),
            )
        else:
            edge_data = {
                "predicates": [predicate],
                "weight": 1,
                "source_texts": [evidence_sentence] if evidence_sentence else [],
                "source_chunk_ids": [source_chunk_id] if source_chunk_id else [],
                "relations": [
                    {
                        "predicate": predicate,
                        "evidence_sentence": evidence_sentence,
                        "source_chunk_id": source_chunk_id,
                    }
                ],
            }
            if description:
                edge_data["description"] = description
            conn.execute(
                "INSERT INTO edges (subject, object, attrs) VALUES (?, ?, ?)",
                (subject, object_id, json.dumps(edge_data)),
            )

    # ── Buffering ────────────────────────────────────────────────

    def _buffer_node(self, data: dict[str, Any]) -> None:
        self._node_buffer.append((data["id"], json.dumps(data)))
        self._node_count += 1
        if len(self._node_buffer) >= self._NODE_FLUSH_THRESHOLD:
            self._flush_nodes()

    def _flush_nodes(self) -> None:
        if not self._node_buffer:
            return
        self._graph._conn.executemany(
            "INSERT OR REPLACE INTO nodes (id, attrs) VALUES (?, ?)",
            self._node_buffer,
        )
        self._node_buffer = []

    def _flush(self) -> None:
        """Flush buffered nodes and commit the single write transaction."""
        self._flush_nodes()
        self._graph._conn.commit()

    # ── Access ───────────────────────────────────────────────────

    @property
    def graph(self) -> SQLiteGraph:
        """The file-backed ``SQLiteGraph`` (flushes pending writes first)."""
        self._flush()
        return self._graph

    @property
    def stats(self) -> dict[str, int]:
        return {"nodes_written": self._node_count, "edges_written": self._edge_count}

    def close(self) -> None:
        """Flush pending writes (the ``SQLiteGraph`` owns its connection)."""
        self._flush()
