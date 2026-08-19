"""Neo4j-backed graph builder — writes nodes and edges directly to Neo4j.

Unlike ``GraphBuilder`` (which constructs a full ``networkx.DiGraph`` in RAM),
this builder streams entities and triples directly into a Neo4j database via
Cypher ``MERGE`` operations.  It is designed for incremental, production-scale
knowledge graph construction where the full graph may not fit in memory.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kglab._shared import Ontology
from kglab.kg_build.build.writer import GraphWriter

logger = logging.getLogger(__name__)

# Characters that are not valid in unquoted Neo4j relationship type identifiers.
_INVALID_REL_TYPE_CHARS = re.compile(r"[^A-Za-z0-9_]")

_ID_CONSTRAINTS = {
    "Entity": "polygraph_entity_id",
    "Chunk": "polygraph_chunk_id",
    "Document": "polygraph_document_id",
}


def _safe_rel_type(predicate: str) -> str:
    """Map an arbitrary predicate string to a safe Neo4j relationship type."""
    cleaned = _INVALID_REL_TYPE_CHARS.sub("_", predicate.upper()).strip("_")
    if not cleaned:
        return "RELATION"
    if cleaned[0].isdigit():
        return f"REL_{cleaned}"
    return cleaned


def ensure_schema(session: Any) -> None:
    """Create stable-ID constraints used by all Neo4j writers."""
    for label, constraint_name in _ID_CONSTRAINTS.items():
        session.run(
            f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (node:{label}) REQUIRE node.id IS UNIQUE
            """
        )


class Neo4jGraphBuilder(GraphWriter):
    """Build a knowledge graph by writing directly to a Neo4j database.

    Implements the shared ``GraphWriter`` interface (see
    ``kglab.kg_build.build.writer``), so the same ``build_kg_into``
    routine drives both this streaming backend and the in-memory
    ``NetworkXGraphWriter``.
    """

    def __init__(
        self,
        session: Any,
        ontology: Ontology | None = None,
    ) -> None:
        """
        Parameters
        ----------
        session:
            An active Neo4j ``Session`` (from ``neo4j.Driver.session()``).
        ontology:
            Optional ontology used for validation logging (not enforced at write-time).
        """
        self.session = session
        self.ontology = ontology
        self._node_count = 0
        self._edge_count = 0
        ensure_schema(self.session)

    # ── Node helpers ───────────────────────────────────────────────

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
        """Create or update a ``:Chunk`` node."""
        self.session.run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.type   = 'Chunk',
                c.source = $source,
                c.text   = $text,
                c.tokenCount = $tokenCount,
                c.index  = $index
            SET c += $properties
            REMOVE c.entityType
            """,
            id=chunk_id,
            source=source,
            text=text,
            tokenCount=token_count,
            index=index,
            properties=properties or {},
        )
        self._node_count += 1

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
        """Create or update a ``:Document`` node."""
        self.session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.name        = $name,
                d.type        = 'Document',
                d.description = $description,
                d.source      = $source,
                d.chunk_count = $chunk_count
            SET d += $properties
            REMOVE d.entityType
            """,
            id=doc_id,
            name=name,
            description=description,
            source=source,
            chunk_count=chunk_count,
            properties=properties or {},
        )
        self._node_count += 1

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
        """Create or update an ``:Entity`` node.

        Extra ``properties`` are merged onto the node (they must not clash
        with the explicitly-set keys above).
        """
        self.session.run(
            """
            MERGE (n:Entity {id: $id})
            SET n.name             = $name,
                n.type             = $type,
                n.description      = $description,
                n.importanceScore  = $importanceScore,
                n.confidenceScore  = $confidenceScore,
                n.embedding        = $embedding
            SET n += $properties
            REMOVE n.entityType
            WITH n
            WHERE $aliases IS NOT NULL AND size($aliases) > 0
            SET n.aliases = [a IN $aliases WHERE NOT a IN coalesce(n.aliases, [])]
                          + coalesce(n.aliases, [])
            """,
            id=entity_id,
            name=name,
            type=entity_type,
            description=description,
            importanceScore=importance_score,
            confidenceScore=confidence_score,
            embedding=embedding,
            aliases=aliases or [],
            properties=properties or {},
        )
        self._node_count += 1

    # ── Edge helpers ───────────────────────────────────────────────

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
        """Create or update a relationship between two nodes.

        If the relationship already exists its ``sourceChunkIds`` and
        ``evidenceSentences`` arrays are extended (deduplicated).
        """
        safe_pred = _safe_rel_type(predicate)

        self.session.run(
            f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{safe_pred}]->(b)
            SET r.weight = coalesce(r.weight, 0) + $weight,
                r.sourceChunkIds = CASE
                    WHEN $source_chunk_id <> '' AND NOT $source_chunk_id IN
                         coalesce(r.sourceChunkIds, [])
                    THEN coalesce(r.sourceChunkIds, []) + [$source_chunk_id]
                    ELSE coalesce(r.sourceChunkIds, [])
                END,
                r.evidenceSentences = CASE
                    WHEN $evidence_sentence <> '' AND NOT $evidence_sentence IN
                         coalesce(r.evidenceSentences, [])
                    THEN coalesce(r.evidenceSentences, []) + [$evidence_sentence]
                    ELSE coalesce(r.evidenceSentences, [])
                END,
                r.description = CASE
                    WHEN $description <> ''
                    THEN $description
                    ELSE coalesce(r.description, '')
                END
            """,
            source=source_id,
            target=target_id,
            weight=weight,
            source_chunk_id=source_chunk_id,
            evidence_sentence=evidence_sentence,
            description=description,
        )
        self._edge_count += 1

    def merge_structural_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> None:
        """Create a simple structural edge (APPEARS_IN, PART_OF, NEXT).

        These edges carry no additional properties beyond their type.
        """
        safe_rel = _safe_rel_type(relationship_type)

        self.session.run(
            f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[:{safe_rel}]->(b)
            """,
            source=source_id,
            target=target_id,
        )
        self._edge_count += 1

    # ── Lifecycle ──────────────────────────────────────────────────

    def remove_document_chunks(self, document_id: str) -> int:
        """Remove all ``:Chunk`` nodes (and their edges) for a given document.

        Uses ``DETACH DELETE`` so that ``APPEARS_IN``, ``NEXT``, and
        ``PART_OF`` relationships are automatically removed. ``:Entity``
        nodes are left untouched because they may be shared across documents.

        Returns the number of chunks removed.
        """
        result = self.session.run(
            """
            MATCH (d:Document {id: $doc_id})
            MATCH (c:Chunk)-[:PART_OF]->(d)
            DETACH DELETE c
            RETURN count(c) AS removed
            """,
            doc_id=document_id,
        )
        count = result.single()["removed"]
        if count:
            logger.info("Removed %d chunk(s) for document %s from Neo4j", count, document_id)
        return count

    def clear_database(self) -> None:
        """Delete all graph data while preserving the stable-ID schema."""
        self.session.run("MATCH (n) DETACH DELETE n")
        ensure_schema(self.session)
        logger.info("Cleared all nodes and relationships from Neo4j")

    @property
    def stats(self) -> dict[str, int]:
        """Running counters of nodes and edges written during this session."""
        return {"nodes_written": self._node_count, "edges_written": self._edge_count}

    # ── Post-processing ────────────────────────────────────────────

    def compute_pagerank(self, graph_name: str = "kg-graph") -> None:
        """Run PageRank via Neo4j GDS and write ``importanceScore`` on every node.

        Requires the Neo4j Graph Data Science library to be installed.
        """
        try:
            # Drop any stale in-memory graph projection
            self.session.run(
                f"CALL gds.graph.exists('{graph_name}') YIELD exists "
                "WITH exists WHERE exists "
                f"CALL gds.graph.drop('{graph_name}') YIELD graphName "
                "RETURN graphName"
            )

            self.session.run(
                f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    ['Entity', 'Chunk', 'Document'],
                    '*'
                )
                """
            )

            self.session.run(
                f"""
                CALL gds.pageRank.write('{graph_name}', {{
                    maxIterations: 100,
                    dampingFactor: 0.85,
                    writeProperty: 'importanceScore'
                }})
                YIELD nodePropertiesWritten, ranIterations
                """
            )

            self.session.run(f"CALL gds.graph.drop('{graph_name}')")
            logger.info("PageRank scores written to Neo4j via GDS")
        except Exception as exc:
            logger.warning("GDS PageRank failed (GDS library may not be installed): %s", exc)

    def compute_stats(self) -> dict[str, Any]:
        """Return graph-level statistics queried directly from Neo4j."""
        result = self.session.run(
            """
            CALL () { MATCH (n) RETURN count(n) AS node_count }
            CALL () { MATCH ()-[r]->() RETURN count(r) AS edge_count }
            RETURN node_count, edge_count
            """
        ).single()

        if result is None:
            return {"num_nodes": 0, "num_edges": 0}

        label_dist = self.session.run(
            """
            MATCH (n)
            WITH labels(n)[0] AS label, count(n) AS cnt
            WHERE label IS NOT NULL
            RETURN label, cnt
            ORDER BY cnt DESC
            """
        ).data()

        return {
            "num_nodes": result["node_count"],
            "num_edges": result["edge_count"],
            "label_distribution": {row["label"]: row["cnt"] for row in label_dist},
        }
