#!/usr/bin/env python3
"""Upload or atomically replace KG1-A.2 through native Cypher."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from kglab.kg_export.neo4j.upload import _get_connection

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = (
    EXPERIMENT_DIR / "artifacts" / "kg1_revision_pilot" / "neo4j_import" / "knowledge_graph.json"
)
DEFAULT_REPORT = EXPERIMENT_DIR / "artifacts" / "kg1_revision_pilot" / "neo4j_confirmation.json"
CURRENT_GRAPH = "KG1-A.2"
REPLACEABLE_GRAPHS = ("KG1-A", CURRENT_GRAPH)

NODE_QUERIES = {
    "ARTICLE": """
        UNWIND $rows AS row
        MERGE (n:Entity:KG1Node:Article {id: row.id})
        SET n += row.properties
    """,
    "CHUNK": """
        UNWIND $rows AS row
        MERGE (n:Entity:KG1Node:Chunk {id: row.id})
        SET n += row.properties
    """,
    "ENTITY": """
        UNWIND $rows AS row
        MERGE (n:Entity:KG1Node:KG1Entity {id: row.id})
        SET n += row.properties
    """,
}
ALLOWED_PREDICATES = {
    "DESCRIBES",
    "HAS_CHUNK",
    "LINKS_TO",
    "MENTIONS",
    "NEXT",
}
PROPERTY_KEYS = {
    "article_id",
    "article_revision_id",
    "chunk_index",
    "chunk_method",
    "confidenceScore",
    "dataset_article_id",
    "description",
    "external_context",
    "importanceScore",
    "kg1NodeType",
    "name",
    "page_id",
    "primary_spacy_type",
    "qid",
    "split",
    "spacyTypeCountsJson",
    "spacy_types",
    "start_char",
    "end_char",
    "text",
    "title",
    "type",
}


def node_properties(node: dict[str, Any]) -> dict[str, Any]:
    properties = {
        key: value for key, value in node.items() if key in PROPERTY_KEYS and value is not None
    }
    if "spacy_type_counts" in node:
        properties["spacyTypeCountsJson"] = json.dumps(
            node["spacy_type_counts"],
            ensure_ascii=False,
            sort_keys=True,
        )
    properties["kg1Graph"] = CURRENT_GRAPH
    return properties


def group_payload(
    data: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in data["graph"]["nodes"]:
        node_type = node["type"]
        if node_type not in NODE_QUERIES:
            raise ValueError(f"Unsupported Neo4j node type: {node_type!r}")
        nodes_by_type[node_type].append({"id": node["id"], "properties": node_properties(node)})

    edges_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in data["graph"]["edges"]:
        for predicate in edge["predicates"]:
            if predicate not in ALLOWED_PREDICATES:
                raise ValueError(f"Unsupported KG1 predicate: {predicate!r}")
            edges_by_type[predicate].append(
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": int(edge.get("weight", 1)),
                }
            )
    return dict(nodes_by_type), dict(edges_by_type)


def relationship_query(predicate: str) -> str:
    if predicate not in ALLOWED_PREDICATES:
        raise ValueError(f"Unsupported KG1 predicate: {predicate!r}")
    return f"""
        UNWIND $rows AS row
        MATCH (source:KG1Node {{id: row.source}})
        MATCH (target:KG1Node {{id: row.target}})
        MERGE (source)-[relationship:{predicate}]->(target)
        SET relationship.kg1Graph = 'KG1-A.2',
            relationship.weight = row.weight
    """


def _read_payload(
    path: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return group_payload(data)


def _write_payload(
    tx: Any,
    nodes_by_type: dict[str, list[dict[str, Any]]],
    edges_by_type: dict[str, list[dict[str, Any]]],
) -> None:
    for node_type, rows in nodes_by_type.items():
        tx.run(NODE_QUERIES[node_type], rows=rows).consume()
    for predicate, rows in edges_by_type.items():
        tx.run(relationship_query(predicate), rows=rows).consume()


def _upload_summary(
    nodes_by_type: dict[str, list[dict[str, Any]]],
    edges_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "nodes": sum(len(rows) for rows in nodes_by_type.values()),
        "relationships": sum(len(rows) for rows in edges_by_type.values()),
        "node_types": dict(
            sorted((node_type, len(rows)) for node_type, rows in nodes_by_type.items())
        ),
        "predicates": dict(
            sorted((predicate, len(rows)) for predicate, rows in edges_by_type.items())
        ),
    }


def upload(path: Path) -> dict[str, Any]:
    load_dotenv()
    nodes_by_type, edges_by_type = _read_payload(path)

    driver = _get_connection()
    try:
        with driver.session() as session:

            def write(tx: Any) -> None:
                _write_payload(tx, nodes_by_type, edges_by_type)

            session.execute_write(write)
    finally:
        driver.close()

    return _upload_summary(nodes_by_type, edges_by_type)


def replace(path: Path) -> dict[str, Any]:
    """Atomically delete KG1 versions and upload the current payload."""
    load_dotenv()
    nodes_by_type, edges_by_type = _read_payload(path)

    driver = _get_connection()
    try:
        with driver.session() as session:

            def write(tx: Any) -> dict[str, int]:
                previous = tx.run(
                    """
                    MATCH (node:KG1Node)
                    WHERE node.kg1Graph IN $graph_versions
                    OPTIONAL MATCH (node)-[relationship]-()
                    RETURN count(DISTINCT node) AS nodes,
                           count(DISTINCT relationship) AS relationships
                    """,
                    graph_versions=list(REPLACEABLE_GRAPHS),
                ).single()
                tx.run(
                    """
                    MATCH (node:KG1Node)
                    WHERE node.kg1Graph IN $graph_versions
                    DETACH DELETE node
                    """,
                    graph_versions=list(REPLACEABLE_GRAPHS),
                ).consume()
                _write_payload(tx, nodes_by_type, edges_by_type)
                return {
                    "nodes": previous["nodes"],
                    "relationships": previous["relationships"],
                }

            cleared = session.execute_write(write)
    finally:
        driver.close()

    return {
        "cleared": cleared,
        "uploaded": _upload_summary(nodes_by_type, edges_by_type),
    }


def inspect(path: Path) -> dict[str, Any]:
    load_dotenv()
    data = json.loads(path.read_text(encoding="utf-8"))
    node_ids = [node["id"] for node in data["graph"]["nodes"]]

    driver = _get_connection()
    try:
        with driver.session() as session:
            node_count = session.run(
                """
                MATCH (node:KG1Node)
                WHERE node.id IN $node_ids
                RETURN count(node) AS count
                """,
                node_ids=node_ids,
            ).single()["count"]
            node_types = {
                record["type"]: record["count"]
                for record in session.run(
                    """
                    MATCH (node:KG1Node)
                    WHERE node.id IN $node_ids
                    RETURN node.type AS type, count(node) AS count
                    ORDER BY type
                    """,
                    node_ids=node_ids,
                )
            }
            relationships = {
                record["predicate"]: record["count"]
                for record in session.run(
                    """
                    MATCH (source:KG1Node)-[relationship]->(target:KG1Node)
                    WHERE source.id IN $node_ids
                      AND target.id IN $node_ids
                      AND relationship.kg1Graph = 'KG1-A.2'
                    RETURN type(relationship) AS predicate,
                           count(relationship) AS count
                    ORDER BY predicate
                    """,
                    node_ids=node_ids,
                )
            }
            weighted_relationships = session.run(
                """
                MATCH (source:KG1Node)-[relationship]->(target:KG1Node)
                WHERE source.id IN $node_ids
                  AND target.id IN $node_ids
                  AND relationship.kg1Graph = 'KG1-A.2'
                RETURN sum(coalesce(relationship.weight, 1)) AS count
                """,
                node_ids=node_ids,
            ).single()["count"]
            graph_inventory = {
                record["graph"]: record["count"]
                for record in session.run(
                    """
                    MATCH (node:KG1Node)
                    WHERE node.kg1Graph IN $graph_versions
                    RETURN node.kg1Graph AS graph, count(node) AS count
                    ORDER BY graph
                    """,
                    graph_versions=list(REPLACEABLE_GRAPHS),
                )
            }
            stale_relationships = session.run(
                """
                MATCH ()-[relationship]->()
                WHERE relationship.kg1Graph IN $stale_graphs
                RETURN count(relationship) AS count
                """,
                stale_graphs=[graph for graph in REPLACEABLE_GRAPHS if graph != CURRENT_GRAPH],
            ).single()["count"]
    finally:
        driver.close()

    stale_nodes = sum(count for graph, count in graph_inventory.items() if graph != CURRENT_GRAPH)
    return {
        "nodes": node_count,
        "relationships": sum(relationships.values()),
        "weighted_relationships": weighted_relationships,
        "node_types": node_types,
        "predicates": relationships,
        "expected_nodes": data["stats"]["num_nodes"],
        "expected_relationships": data["stats"]["num_edges"],
        "expected_weighted_relationships": data["stats"]["edge_weight_sum"],
        "graph_inventory": graph_inventory,
        "stale_nodes": stale_nodes,
        "stale_relationships": stale_relationships,
        "confirmed": (
            node_count == data["stats"]["num_nodes"]
            and sum(relationships.values()) == data["stats"]["num_edges"]
            and weighted_relationships == data["stats"]["edge_weight_sum"]
            and Counter(relationships) == Counter(data["stats"]["predicates"])
            and graph_inventory.get(CURRENT_GRAPH) == data["stats"]["num_nodes"]
            and stale_nodes == 0
            and stale_relationships == 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("upload", "replace", "inspect"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "upload":
        result = upload(args.input)
    elif args.command == "replace":
        result = replace(args.input)
    else:
        result = inspect(args.input)
    if args.command == "inspect":
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
