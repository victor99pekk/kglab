"""Upload/download a knowledge graph to/from a Neo4j database."""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from polygraph.kg_export.neo4j.builder import _safe_rel_type, ensure_schema

logger = logging.getLogger(__name__)

_STRUCTURAL_RELATIONSHIPS = {"APPEARS_IN", "PART_OF", "NEXT"}


def replace_documents(session, document_ids: list[str]) -> None:
    """Delete existing KG data owned by incoming document IDs.

    Document IDs are stable across content changes, while chunk IDs include the
    chunk text. Removing the old chunks before upload prevents changed documents
    from leaving stale chunks and extracted relationships in Neo4j.
    """
    document_ids = sorted({document_id for document_id in document_ids if document_id})
    if not document_ids:
        return

    record = session.run(
        """
        MATCH (chunk:Chunk)-[:PART_OF]->(document:Document)
        WHERE document.id IN $document_ids
        OPTIONAL MATCH (entity:Entity)-[:APPEARS_IN]->(chunk)
        RETURN collect(DISTINCT chunk.id) AS chunk_ids,
               collect(DISTINCT entity.id) AS entity_ids
        """,
        document_ids=document_ids,
    ).single()
    chunk_ids = list(record["chunk_ids"] or []) if record else []
    entity_ids = list(record["entity_ids"] or []) if record else []

    if chunk_ids:
        # Extracted entity-to-entity relationships are not directly connected
        # to Chunk nodes. Remove replaced chunk IDs from their provenance and
        # delete a relationship only when no other document still supports it.
        # Evidence is retained when other chunks still support the relationship;
        # the current schema cannot safely assign each sentence to one chunk.
        session.run(
            """
            MATCH ()-[relationship]->()
            WHERE NOT type(relationship) IN ['APPEARS_IN', 'PART_OF', 'NEXT']
              AND any(chunk_id IN coalesce(relationship.sourceChunkIds, [])
                      WHERE chunk_id IN $chunk_ids)
            WITH relationship,
                 [chunk_id IN coalesce(relationship.sourceChunkIds, [])
                  WHERE NOT chunk_id IN $chunk_ids] AS remaining_chunk_ids
            SET relationship.sourceChunkIds = remaining_chunk_ids
            WITH relationship, remaining_chunk_ids
            WHERE size(remaining_chunk_ids) = 0
            DELETE relationship
            """,
            chunk_ids=chunk_ids,
        )

    session.run(
        """
        MATCH (chunk:Chunk)-[:PART_OF]->(document:Document)
        WHERE document.id IN $document_ids
        DETACH DELETE chunk
        """,
        document_ids=document_ids,
    )
    session.run(
        """
        MATCH (document:Document)
        WHERE document.id IN $document_ids
        DETACH DELETE document
        """,
        document_ids=document_ids,
    )

    if entity_ids:
        # Remove entities owned only by replaced documents. Entities still
        # mentioned by an unrelated document remain in the graph.
        session.run(
            """
            MATCH (entity:Entity)
            WHERE entity.id IN $entity_ids
              AND NOT EXISTS { MATCH (entity)-[:APPEARS_IN]->(:Chunk) }
            DETACH DELETE entity
            """,
            entity_ids=entity_ids,
        )


def replace_documents_atomic(
    session,
    document_ids: list[str],
    writer,
    *,
    clear_all: bool = False,
) -> None:
    """Replace documents and write new data in one Neo4j transaction.

    ``writer`` receives the transaction object and may use ``tx.run`` or pass
    it to the streaming graph builder.  Any exception rolls back both deletion
    and writes, leaving the previous graph intact.
    """

    def work(tx):
        if clear_all:
            tx.run("MATCH (n) DETACH DELETE n")
        else:
            replace_documents(tx, document_ids)
        writer(tx)

    if hasattr(session, "execute_write"):
        session.execute_write(work)
    elif hasattr(session, "write_transaction"):
        session.write_transaction(work)
    else:
        # Small fake sessions and older drivers can expose an explicit tx API.
        tx = session.begin_transaction()
        try:
            work(tx)
            tx.commit()
        except Exception:
            tx.rollback()
            raise


def _get_connection(
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
):
    """Create a Neo4j driver using explicit values or environment defaults."""
    from neo4j import GraphDatabase

    uri = uri or os.environ.get("NEO4J_URI", "")
    user = user or os.environ.get("NEO4J_USER", "")
    password = password if password is not None else os.environ.get("NEO4J_PASSWORD", "")

    if not uri or not user or not password:
        raise RuntimeError(
            "Neo4j credentials not set. Ensure NEO4J_URI, NEO4J_USER, "
            "and NEO4J_PASSWORD are defined in .env"
        )

    return GraphDatabase.driver(uri, auth=(user, password))


def clear_database(
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """Delete all nodes and relationships from the Neo4j database."""
    driver = _get_connection(uri=uri, user=user, password=password)
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
        logger.info("Cleared all nodes and relationships from Neo4j")
    driver.close()


def _canonical_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical entity records, enriched by graph-only properties.

    Current exports contain both resolved ``entities`` and derived graph nodes.
    Entity records are authoritative; graph nodes provide computed properties
    such as PageRank and backward-compatible endpoint placeholders.
    """
    graph_nodes = data.get("graph", {}).get("nodes", [])
    graph_by_id = {
        node["id"]: dict(node)
        for node in graph_nodes
        if isinstance(node, dict) and node.get("id")
    }
    if "entities" not in data:
        return list(graph_by_id.values())

    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in data.get("entities", []):
        if not isinstance(entity, dict) or not entity.get("id"):
            continue
        node_id = str(entity["id"])
        graph_node = graph_by_id.get(node_id, {})
        merged = {**graph_node, **entity, "id": node_id}
        if "importanceScore" in graph_node:
            merged["importanceScore"] = graph_node["importanceScore"]
        canonical.append(merged)
        seen.add(node_id)

    canonical.extend(node for node_id, node in graph_by_id.items() if node_id not in seen)
    return canonical


def _triple_record(triple: Any) -> dict[str, Any]:
    """Normalize one exported triple dict or positional record."""
    if isinstance(triple, dict):
        record = {
            "source": triple.get("subject", triple.get("source", "")),
            "predicate": triple.get("predicate", triple.get("type", "")),
            "target": triple.get("object", triple.get("target", "")),
            "evidence_sentence": triple.get("evidence_sentence", ""),
            "source_chunk_id": triple.get("source_chunk_id", ""),
            "description": triple.get("description", ""),
        }
        if triple.get("source_chunk_ids"):
            record["source_chunk_ids"] = triple["source_chunk_ids"]
        return record

    if not isinstance(triple, (list, tuple)) or len(triple) < 3:
        raise ValueError(f"Invalid triple record: {triple!r}")
    return {
        "source": triple[0],
        "predicate": triple[1],
        "target": triple[2],
        "evidence_sentence": triple[3] if len(triple) > 3 else "",
        "source_chunk_id": triple[4] if len(triple) > 4 else "",
        "description": triple[5] if len(triple) > 5 else "",
    }


def _legacy_edge_records(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover individual relationship records from legacy graph edges."""
    records: list[dict[str, Any]] = []
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        predicates = edge.get("predicates") or [edge.get("type", "related_to")]
        relation_records = edge.get("relations", [])

        for predicate in predicates:
            matching = [
                relation
                for relation in relation_records
                if relation.get("predicate") == predicate
            ]
            if not matching:
                records.append(
                    {
                        "source": source,
                        "predicate": predicate,
                        "target": target,
                        "evidence_sentence": "",
                        "source_chunk_ids": edge.get("source_chunk_ids", []),
                        "description": edge.get("description", ""),
                    }
                )
                continue

            for relation in matching:
                records.append(
                    {
                        "source": source,
                        "predicate": predicate,
                        "target": target,
                        "evidence_sentence": relation.get("evidence_sentence", ""),
                        "source_chunk_id": relation.get("source_chunk_id", ""),
                        "description": relation.get("description", ""),
                    }
                )
    return records


def _canonical_relationships(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer canonical triples and fall back to legacy graph edges."""
    if "triples" in data:
        return [_triple_record(triple) for triple in data.get("triples", [])]
    return _legacy_edge_records(data.get("graph", {}).get("edges", []))


def _validate_endpoints(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    """Reject missing endpoints and structurally invalid relationships."""
    node_types = {
        str(node["id"]): str(node.get("type", "Entity"))
        for node in nodes
        if node.get("id")
    }
    node_ids = set(node_types)
    missing = sorted(
        {
            str(endpoint)
            for relationship in relationships
            for endpoint in (relationship.get("source", ""), relationship.get("target", ""))
            if endpoint and str(endpoint) not in node_ids
        }
    )
    if missing:
        preview = ", ".join(missing[:5])
        remainder = len(missing) - 5
        if remainder > 0:
            preview += f", and {remainder} more"
        raise ValueError(f"Relationship endpoint(s) missing from node records: {preview}")

    expected_types = {
        "APPEARS_IN": ("Entity", "Chunk"),
        "PART_OF": ("Chunk", "Document"),
        "NEXT": ("Chunk", "Chunk"),
    }
    violations: list[str] = []
    for relationship in relationships:
        predicate = _safe_rel_type(str(relationship.get("predicate", "")))
        expected = expected_types.get(predicate)
        if expected is None:
            continue
        source = str(relationship.get("source", ""))
        target = str(relationship.get("target", ""))
        source_type = node_types.get(source, "")
        target_type = node_types.get(target, "")
        source_matches = (
            expected[0] == "Entity" and source_type not in {"Chunk", "Document"}
        ) or source_type == expected[0]
        if not source_matches or target_type != expected[1]:
            violations.append(
                f"{predicate}: expected {expected[0]}->{expected[1]}, "
                f"got {source_type}->{target_type} ({source}->{target})"
            )
    if violations:
        raise ValueError(f"Invalid structural relationship(s): {'; '.join(violations[:5])}")


def upload_graph(
    json_path: str | Path,
    clear: bool = False,
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> None:
    """Upload canonical entities and triples from a KG export.

    Older exports without top-level ``entities`` or ``triples`` fall back to
    their derived graph nodes and edges. Connection arguments override the
    corresponding ``NEO4J_*`` environment variables when provided.

    Args:
        json_path: Path to ``knowledge_graph.json``.
        clear: If True, wipe the database before uploading.
        uri: Neo4j bolt URI (overrides ``NEO4J_URI`` env var).
        user: Neo4j username (overrides ``NEO4J_USER`` env var).
        password: Neo4j password (overrides ``NEO4J_PASSWORD`` env var).
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Graph file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    nodes = _canonical_nodes(data)
    relationships = _canonical_relationships(data)
    _validate_endpoints(nodes, relationships)

    driver = _get_connection(uri=uri, user=user, password=password)
    # Build a node-type lookup so relationship MATCH can use label-specific indexes
    id_to_type: dict[str, str] = {
        node["id"]: node.get("type", "Entity") for node in nodes if "id" in node
    }

    try:
        with driver.session() as session:
            if clear:
                logger.info("Clearing existing Neo4j data...")
                session.run("MATCH (n) DETACH DELETE n")
                ensure_schema(session)
            else:
                ensure_schema(session)
                replace_documents(
                    session,
                    [node.get("id", "") for node in nodes if node.get("type") == "Document"],
                )

            node_rows = _categorize_nodes(nodes)
            for label in ("Chunk", "Document", "Entity"):
                _upload_nodes_batched(session, label, node_rows[label])

            logger.info("Uploading %d relationship records...", len(relationships))
            _upload_relationships_batched(session, relationships, id_to_type)
    finally:
        driver.close()

    logger.info(
        "Successfully uploaded %d nodes and %d relationship records to Neo4j",
        len(nodes),
        len(relationships),
    )


BATCH_SIZE = 5000
BATCH_SIZE_REL = 20000

_PROVENANCE_KEYS = (
    "title",
    "url",
    "license",
    "source_domain",
    "scraped_at",
    "crawler",
    "content_hash",
    "inferred_type",
)


def _categorize_nodes(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize canonical node records into label-specific upload rows."""
    categorized: dict[str, list[dict[str, Any]]] = {
        "Chunk": [],
        "Document": [],
        "Entity": [],
    }
    for node in nodes:
        node_type = node.get("type", "Entity")
        provenance = {
            key: node[key]
            for key in _PROVENANCE_KEYS
            if node.get(key) not in (None, "")
        }

        if node_type == "Chunk":
            source = node.get("source", "")
            if isinstance(source, list):
                source = source[0] if source else ""
            categorized["Chunk"].append(
                {
                    "id": node.get("id", ""),
                    "source": source,
                    "text": node.get("text", ""),
                    "tokenCount": node.get("tokenCount", 0),
                    "index": node.get("index", 0),
                    "type": "Chunk",
                    "upload_date": node.get("upload_date", ""),
                    "provenance": provenance,
                }
            )
        elif node_type == "Document":
            categorized["Document"].append(
                {
                    "id": node.get("id", ""),
                    "name": node.get("name", ""),
                    "type": "Document",
                    "description": node.get("description", ""),
                    "source": node.get("source", []),
                    "chunk_count": node.get("chunk_count", 0),
                    "upload_date": node.get("upload_date", ""),
                    "provenance": provenance,
                }
            )
        else:
            aliases = node.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases] if aliases else []
            elif not isinstance(aliases, list):
                aliases = list(aliases) if aliases else []
            categorized["Entity"].append(
                {
                    "id": node.get("id", ""),
                    "name": node.get("name", ""),
                    "type": node_type,
                    "aliases": aliases,
                    "description": node.get("description", ""),
                    "importance_score": node.get("importanceScore", 0.0),
                    "confidence_score": node.get("confidenceScore", 1.0),
                    "embedding": node.get("embedding"),
                    "provenance": provenance,
                }
            )
    return categorized


def _upload_nodes_batched(session, label: str, rows: list[dict]) -> None:
    """Upload nodes of a single label in UNWIND batches."""
    if not rows:
        return
    logger.info(f"  Uploading {len(rows)} {label} nodes...")
    for offset in range(0, len(rows), BATCH_SIZE):
        batch = rows[offset : offset + BATCH_SIZE]
        if label == "Chunk":
            session.run(
                """
                UNWIND $batch AS item
                MERGE (n:Chunk {id: item.id})
                SET n.source = item.source, n.text = item.text,
                    n.tokenCount = item.tokenCount, n.index = item.index,
                    n.type = item.type, n.upload_date = item.upload_date
                SET n += item.provenance
                REMOVE n.entityType
                """,
                batch=batch,
            )
        elif label == "Document":
            session.run(
                """
                UNWIND $batch AS item
                MERGE (n:Document {id: item.id})
                SET n.name = item.name, n.type = item.type,
                    n.description = item.description, n.source = item.source,
                    n.chunk_count = item.chunk_count,
                    n.upload_date = item.upload_date
                SET n += item.provenance
                REMOVE n.entityType
                """,
                batch=batch,
            )
        elif label == "Entity":
            session.run(
                """
                UNWIND $batch AS item
                MERGE (n:Entity {id: item.id})
                SET n.name = item.name, n.type = item.type,
                    n.description = CASE WHEN item.description <> ''
                        THEN item.description ELSE coalesce(n.description, '') END,
                    n.importanceScore = item.importance_score,
                    n.confidenceScore = CASE
                        WHEN item.confidence_score > coalesce(n.confidenceScore, 0)
                        THEN item.confidence_score
                        ELSE coalesce(n.confidenceScore, 0) END,
                    n.embedding = coalesce(item.embedding, n.embedding),
                    n.aliases = reduce(
                        aliases = coalesce(n.aliases, []),
                        alias IN item.aliases |
                        CASE WHEN alias IN aliases THEN aliases ELSE aliases + [alias] END
                    )
                SET n += item.provenance
                REMOVE n.entityType
                """,
                batch=batch,
            )


# Label-filtered relationship patterns — one Cypher query per (src_label, tgt_label) combo
# so that Neo4j can use label-specific indexes on :Chunk(id), :Document(id), :Entity(id).
_LABEL_QUERIES: dict[tuple[str, str], str] = {
    ("Chunk", "Chunk"): "MATCH (a:Chunk {id: row.source}), (b:Chunk {id: row.target})",
    ("Chunk", "Document"): "MATCH (a:Chunk {id: row.source}), (b:Document {id: row.target})",
    ("Chunk", "Entity"): "MATCH (a:Chunk {id: row.source}), (b:Entity {id: row.target})",
    ("Document", "Chunk"): "MATCH (a:Document {id: row.source}), (b:Chunk {id: row.target})",
    ("Document", "Document"): "MATCH (a:Document {id: row.source}), (b:Document {id: row.target})",
    ("Document", "Entity"): "MATCH (a:Document {id: row.source}), (b:Entity {id: row.target})",
    ("Entity", "Chunk"): "MATCH (a:Entity {id: row.source}), (b:Chunk {id: row.target})",
    ("Entity", "Document"): "MATCH (a:Entity {id: row.source}), (b:Document {id: row.target})",
    ("Entity", "Entity"): "MATCH (a:Entity {id: row.source}), (b:Entity {id: row.target})",
}


def _neo4j_label(node_type: str) -> str:
    """Map a node type to the Neo4j label used at upload time.

    Chunk and Document keep their labels; all entity subtypes (PERSON, ORG, etc.)
    are stored under the ``Entity`` label.
    """
    if node_type in ("Chunk", "Document"):
        return node_type
    return "Entity"


def _relationship_buckets(
    relationships: list[dict[str, Any]],
    id_to_type: dict[str, str],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Aggregate one bounded relationship batch by endpoints and predicate."""
    aggregated: dict[tuple[str, str, str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {
            "descriptions": set(),
            "evidence_sentences": set(),
            "source_chunk_ids": set(),
        }
    )
    for relationship in relationships:
        source = str(relationship.get("source", ""))
        target = str(relationship.get("target", ""))
        predicate = _safe_rel_type(str(relationship.get("predicate", "")))
        if not source or not target:
            continue

        source_label = _neo4j_label(id_to_type.get(source, "Entity"))
        target_label = _neo4j_label(id_to_type.get(target, "Entity"))
        key = (source, target, predicate, source_label, target_label)
        metadata = aggregated[key]

        if predicate not in _STRUCTURAL_RELATIONSHIPS:
            source_chunk_ids = relationship.get("source_chunk_ids", [])
            if isinstance(source_chunk_ids, str):
                source_chunk_ids = [source_chunk_ids]
            elif not isinstance(source_chunk_ids, (list, tuple, set)):
                source_chunk_ids = [source_chunk_ids] if source_chunk_ids else []
            source_chunk_id = relationship.get("source_chunk_id", "")
            if source_chunk_id:
                source_chunk_ids = [*source_chunk_ids, source_chunk_id]
            metadata["source_chunk_ids"].update(
                str(chunk_id) for chunk_id in source_chunk_ids if chunk_id
            )
            evidence = relationship.get("evidence_sentence", "")
            if evidence:
                metadata["evidence_sentences"].add(str(evidence))
            description = relationship.get("description", "")
            if description:
                metadata["descriptions"].add(str(description))

    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (source, target, predicate, source_label, target_label), metadata in aggregated.items():
        buckets[(source_label, target_label, predicate)].append(
            {
                "source": source,
                "target": target,
                "evidence_sentences": sorted(metadata["evidence_sentences"]),
                "source_chunk_ids": sorted(metadata["source_chunk_ids"]),
                "description": min(metadata["descriptions"], default=""),
            }
        )
    return buckets


def _upload_relationships_batched(
    session,
    relationships: list[dict[str, Any]],
    id_to_type: dict[str, str],
) -> None:
    """Upload bounded canonical triple batches without requiring APOC."""
    for offset in range(0, len(relationships), BATCH_SIZE_REL):
        input_batch = relationships[offset : offset + BATCH_SIZE_REL]
        buckets = _relationship_buckets(input_batch, id_to_type)

        for (source_label, target_label, predicate), rows in buckets.items():
            match_clause = _LABEL_QUERIES[(source_label, target_label)]
            if predicate in _STRUCTURAL_RELATIONSHIPS:
                query = f"""
                UNWIND $rows AS row
                {match_clause}
                MERGE (a)-[rel:{predicate}]->(b)
                RETURN count(rel)
                """
            else:
                query = f"""
                UNWIND $rows AS row
                {match_clause}
                MERGE (a)-[rel:{predicate}]->(b)
                SET rel.evidenceSentences = reduce(
                        evidence = coalesce(rel.evidenceSentences, []),
                        sentence IN row.evidence_sentences |
                        CASE WHEN sentence IN evidence
                            THEN evidence ELSE evidence + [sentence] END
                    ),
                    rel.sourceChunkIds = reduce(
                        chunks = coalesce(rel.sourceChunkIds, []),
                        chunk_id IN row.source_chunk_ids |
                        CASE WHEN chunk_id IN chunks
                            THEN chunks ELSE chunks + [chunk_id] END
                    ),
                    rel.description = CASE WHEN row.description <> ''
                        THEN row.description ELSE coalesce(rel.description, '') END
                RETURN count(rel)
                """
            session.run(query, rows=rows)


def upload_from_output(output_dir: str | Path, clear: bool = False) -> None:
    """Upload a knowledge graph from a pipeline output directory to Neo4j."""
    output_dir = Path(output_dir)
    json_path = output_dir / "knowledge_graph.json"

    if not json_path.exists():
        # Try finding in neo4j_import subfolder
        json_path = output_dir / "neo4j_import" / "knowledge_graph.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"No knowledge_graph.json found in {output_dir}. "
            f"Run the pipeline with 'json' in export_formats first."
        )

    upload_graph(json_path, clear=clear)


def download_graph(output_path: str | Path) -> None:
    """Download the full knowledge graph from Neo4j and save as knowledge_graph.json.

    Reconstructs the format expected by the evaluation pipeline
    (kg_generator/evaluate/run_eval.py) from Neo4j. Reads connection details from
    .env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).

    Use case: generate KG locally → upload to Neo4j → download on Colab for eval.
    """
    output_path = Path(output_path)
    driver = _get_connection()

    with driver.session() as session:
        # ── Fetch Entity nodes ──
        entity_result = session.run(
            "MATCH (n:Entity) "
            "RETURN n.id AS id, n.type AS type, "
            "n.description AS description, n.importanceScore AS importanceScore, "
            "n.confidenceScore AS confidenceScore, n.embedding AS embedding"
        )
        entity_nodes = [dict(record) for record in entity_result]

        # ── Fetch Document nodes ──
        doc_result = session.run(
            "MATCH (d:Document) "
            "RETURN d.id AS id, d.name AS name, d.type AS type, "
            "d.description AS description, d.source AS source, "
            "d.chunk_count AS chunk_count"
        )
        doc_nodes = [dict(record) for record in doc_result]

        # ── Fetch Chunk nodes ──
        chunk_result = session.run(
            "MATCH (c:Chunk) "
            "RETURN c.id AS id, c.source AS source, c.text AS text, "
            "c.tokenCount AS tokenCount, c.index AS index"
        )
        chunk_nodes = [dict(record) for record in chunk_result]

        # ── Fetch relationships ──
        rel_result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN a.id AS source, b.id AS target, type(r) AS predicate, "
            "r.weight AS weight"
        )
        relationships = [dict(record) for record in rel_result]

    driver.close()

    if not entity_nodes and not doc_nodes and not chunk_nodes:
        raise RuntimeError(
            "Neo4j database is empty — no nodes found. Upload a graph first: make upload"
        )

    # ── Reconstruct knowledge_graph.json ──────────────────────

    # Build graph.nodes — one node dict per Neo4j node
    graph_nodes: list[dict] = []

    for en in entity_nodes:
        node: dict = {
            "id": en["id"],
            "type": en.get("type", "Entity"),
            "description": en.get("description", ""),
            "importanceScore": en.get("importanceScore", 0.0),
            "confidenceScore": en.get("confidenceScore", 1.0),
        }
        if en.get("embedding") is not None:
            node["embedding"] = en["embedding"]
        graph_nodes.append(node)

    for dn in doc_nodes:
        graph_nodes.append(
            {
                "id": dn["id"],
                "type": dn.get("type", "Document"),
                "name": dn.get("name", dn["id"]),
                "description": dn.get("description", ""),
                "source": dn.get("source", ""),
                "chunk_count": dn.get("chunk_count", 0),
            }
        )

    for cn in chunk_nodes:
        graph_nodes.append(
            {
                "id": cn["id"],
                "type": "Chunk",
                "source": cn.get("source", ""),
                "text": cn.get("text", ""),
                "tokenCount": cn.get("tokenCount", 0),
                "index": cn.get("index", 0),
            }
        )

    # Build graph.links — group by (source, target) to collect predicates
    links_map: dict[tuple[str, str], list[str]] = {}
    for rel in relationships:
        key = (rel["source"], rel["target"])
        if key not in links_map:
            links_map[key] = []
        links_map[key].append(rel["predicate"])

    graph_links = [
        {"source": src, "target": tgt, "predicates": preds}
        for (src, tgt), preds in links_map.items()
    ]

    # Build entities list (Entity-type nodes only; used by evaluation code)
    entities = []
    for en in entity_nodes:
        entities.append(
            {
                "name": en["id"],
                "type": en.get("type", "Entity"),
                "description": en.get("description", ""),
            }
        )

    # Build triples list — deduplicate on (subj, pred, obj)
    seen_triples: set[tuple[str, str, str]] = set()
    triples: list[list[str]] = []
    for rel in relationships:
        key = (rel["source"], rel["predicate"], rel["target"])
        if key not in seen_triples:
            seen_triples.add(key)
            triples.append([rel["source"], rel["predicate"], rel["target"], ""])

    # Assemble final output
    output = {
        "graph": {
            "nodes": graph_nodes,
            "links": graph_links,
        },
        "entities": entities,
        "triples": triples,
        "stats": {
            "num_nodes": len(graph_nodes),
            "num_edges": len(graph_links),
            "num_triples": len(triples),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        "Downloaded %d nodes, %d edges, %d triples → %s",
        len(graph_nodes),
        len(graph_links),
        len(triples),
        output_path,
    )
