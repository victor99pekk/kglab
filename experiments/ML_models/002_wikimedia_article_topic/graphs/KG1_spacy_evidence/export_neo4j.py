#!/usr/bin/env python3
"""Adapt a KG1 graph to the repository Neo4j uploader contract."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE = (
    EXPERIMENT_DIR / "artifacts" / "kg1_revision_pilot" / "gnn_projection.json"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR
    / "artifacts"
    / "kg1_revision_pilot"
    / "neo4j_import"
    / "knowledge_graph.json"
)

NODE_TYPE_MAP = {
    "article": "ARTICLE",
    "entity": "ENTITY",
    "entity_type": "ENTITY_TYPE",
}
RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def adapt_node(node: dict[str, Any]) -> dict[str, Any]:
    source_type = node["type"]
    if source_type not in NODE_TYPE_MAP:
        raise ValueError(f"Unsupported KG1 GNN node type: {source_type!r}")

    output = {
        "id": node["id"],
        "name": node.get("title") or node.get("name") or node["id"],
        "type": NODE_TYPE_MAP[source_type],
        "description": node.get("description", ""),
        "importanceScore": node.get("importanceScore", 0.0),
        "confidenceScore": node.get("confidence", 1.0),
        "embedding": node.get("embedding"),
        "kg1NodeType": source_type,
    }
    for key in (
        "article_revision_id",
        "dataset_article_id",
        "external_context",
        "page_id",
        "qid",
        "source",
        "split",
        "title",
    ):
        if key in node:
            output[key] = node[key]
    return output


def relation_type(relation: str) -> str:
    normalized = relation.upper()
    if not RELATION_RE.fullmatch(normalized):
        raise ValueError(f"Unsafe Neo4j relationship type: {relation!r}")
    if normalized == "HAS_TOPIC":
        raise ValueError("HAS_TOPIC cannot enter KG1 message-passing graph")
    return normalized


def adapt_edge(edge: dict[str, Any]) -> dict[str, Any]:
    predicate = relation_type(edge["relation"])
    relation_record = {
        "predicate": predicate,
        "evidence_sentence": edge.get("evidence_sentence", ""),
        "source_chunk_id": edge.get("evidence_id", ""),
        "description": edge.get("description", ""),
    }
    return {
        "source": edge["source"],
        "target": edge["target"],
        "predicates": [predicate],
        "relations": [relation_record],
    }


def adapt_graph(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("labels_as_message_edges") is not False:
        raise ValueError("KG1 source must explicitly disable label message edges")

    nodes = [adapt_node(node) for node in source["nodes"]]
    node_ids = {node["id"] for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("Duplicate node IDs in KG1 projection")

    edges = [adapt_edge(edge) for edge in source["edges"]]
    missing = sorted(
        {
            endpoint
            for edge in edges
            for endpoint in (edge["source"], edge["target"])
            if endpoint not in node_ids
        }
    )
    if missing:
        raise ValueError(f"Edges reference missing nodes: {missing[:5]}")

    predicates = Counter(
        predicate
        for edge in edges
        for predicate in edge["predicates"]
    )
    return {
        "metadata": {
            "experiment": "002_wikimedia_article_topic",
            "graph": "KG1-A",
            "source_schema_version": source.get("schema_version", ""),
            "labels_as_message_edges": False,
            "adapter": "kg1_repository_neo4j_v1",
        },
        "graph": {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": nodes,
            "edges": edges,
        },
        "entities": [
            node for node in nodes if node["type"] in {"ENTITY", "ENTITY_TYPE"}
        ],
        "triples": [
            {
                "subject": edge["source"],
                "predicate": edge["predicates"][0],
                "object": edge["target"],
                "evidence_sentence": "",
                "source_chunk_id": "",
                "description": "",
            }
            for edge in edges
        ],
        "stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_triples": len(edges),
            "node_types": dict(sorted(Counter(node["type"] for node in nodes).items())),
            "predicates": dict(sorted(predicates.items())),
        },
    }


def export(source_path: Path, output_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    adapted = adapt_graph(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(adapted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return adapted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapted = export(args.source, args.output)
    print(json.dumps(adapted["stats"], indent=2))
    print(f"Neo4j input: {args.output}")


if __name__ == "__main__":
    main()

