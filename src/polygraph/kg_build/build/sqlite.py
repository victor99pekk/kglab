"""SQLite graph-construction method — builds a file-backed graph.

Nodes and edges are written to an SQLite database file instead of held in
memory. The returned ``SQLiteGraph`` object mimics ``nx.DiGraph`` enough for
evaluation and export to work transparently.

Usage::

    from polygraph.kg_build.build import from_resolved
    graph = from_resolved(entities, triples, method="sqlite")
    graph.number_of_nodes()  # → SQL query, not RAM count
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from polygraph._shared import GraphBackend, Ontology, entity_id

logger = logging.getLogger(__name__)

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


# ── Builder ────────────────────────────────────────────────────


class SQLiteGraphBuilder:
    """Build a file-backed knowledge graph using SQLite.

    Args:
        db_path: Where to write the ``.db`` file. Defaults to
            ``output_dir/knowledge_graph.db`` when called via the pipeline.
        ontology: Optional ontology for structural validation.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        ontology: Ontology | None = None,
        backend: GraphBackend | None = None,
    ) -> None:
        self.db_path = str(db_path) if db_path else "knowledge_graph.db"
        self.ontology = ontology

    def build(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, ...]],
    ) -> SQLiteGraph:
        """Build a SQLite-backed graph from entities and triples.

        Returns a ``SQLiteGraph`` that can be passed to evaluators and
        exporters just like an ``nx.DiGraph``.
        """
        graph = SQLiteGraph(self.db_path)
        conn = graph._conn

        # Create tables
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

        entity_types: dict[str, str] = {}

        # Insert nodes
        for entity in entities:
            node_type = entity.get("type", entity.get("label", "ENTITY"))
            node_id = entity.get("id") or entity_id(node_type, entity.get("name", ""))
            entity_types[node_id] = node_type
            node_data = {
                "id": node_id,
                "name": entity.get("name", ""),
                "type": node_type,
                "aliases": entity.get("aliases", []),
                "description": entity.get("description", ""),
                "importanceScore": entity.get("importanceScore", 0.0),
                "confidenceScore": entity.get("confidenceScore", 1.0),
                "source": entity.get("source", []),
                "source_chunk_ids": entity.get("source_chunk_ids", []),
                "embedding": entity.get("embedding"),
                "updatedAt": entity.get("updatedAt", ""),
                "text": entity.get("text", ""),
                "tokenCount": entity.get("tokenCount", 0),
                "index": entity.get("index", 0),
                "chunk_count": entity.get("chunk_count", 0),
                "upload_date": entity.get("upload_date", ""),
            }
            for key in (
                "title",
                "url",
                "license",
                "source_domain",
                "scraped_at",
                "crawler",
                "content_hash",
                "inferred_type",
            ):
                if entity.get(key) not in (None, ""):
                    node_data[key] = entity[key]
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, attrs) VALUES (?, ?)",
                (node_id, json.dumps(node_data)),
            )

        # Insert edges
        for triple in triples:
            subject, predicate, object_id = triple[0], triple[1], triple[2]
            evidence_sentence = triple[3] if len(triple) > 3 else ""
            source_chunk_id = triple[4] if len(triple) > 4 else ""
            description = triple[5] if len(triple) > 5 else ""

            # Ensure both endpoints exist
            for nid in (subject, object_id):
                conn.execute(
                    "INSERT OR IGNORE INTO nodes (id, attrs) VALUES (?, ?)",
                    (nid, json.dumps({"id": nid, "type": "ENTITY", "name": nid})),
                )

            # Merge with existing edge if present
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

        conn.commit()

        logger.info(
            "Built SQLite graph: %d nodes, %d edges → %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self.db_path,
        )
        return graph
