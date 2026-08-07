"""Tests for the storage-agnostic graph writer and shared build routine.

The point of this abstraction: one ``build_kg_into`` call drives any
``GraphWriter`` backend — an in-memory ``nx.DiGraph`` (``NetworkXGraphWriter``)
or a Neo4j stream (``Neo4jGraphBuilder``).
"""

import networkx as nx

from polygraph._shared import Document
from polygraph.kg_build.build import (
    GraphBuilder,
    GraphWriter,
    NetworkXGraphWriter,
    build_kg_into,
)
from polygraph.kg_export.neo4j.builder import Neo4jGraphBuilder


def test_networkx_writer_matches_batch_graph_builder():
    """Incremental writes must produce the same graph as the batch builder."""
    entities = [
        {"id": "e1", "name": "Einstein", "type": "PERSON", "description": "physicist"},
        {"id": "e2", "name": "Princeton", "type": "ORG"},
    ]
    triples = [("e1", "worked_at", "e2", "evidence one", "chunk1")]

    expected = GraphBuilder().build(entities, triples)

    writer = NetworkXGraphWriter()
    writer.merge_entity("e1", name="Einstein", entity_type="PERSON", description="physicist")
    writer.merge_entity("e2", name="Princeton", entity_type="ORG")
    writer.merge_edge(
        "e1", "e2", "worked_at", evidence_sentence="evidence one", source_chunk_id="chunk1"
    )

    got = writer.graph

    assert isinstance(got, nx.DiGraph)
    assert got.number_of_nodes() == expected.number_of_nodes() == 2
    assert got.number_of_edges() == expected.number_of_edges() == 1
    assert got.nodes["e1"]["type"] == "PERSON"
    assert got.nodes["e1"]["description"] == "physicist"
    assert got.edges["e1", "e2"]["predicates"] == ["worked_at"]
    assert got.edges["e1", "e2"]["source_texts"] == ["evidence one"]
    assert got.edges["e1", "e2"]["source_chunk_ids"] == ["chunk1"]


def test_neo4j_builder_implements_graph_writer():
    """Neo4jGraphBuilder must satisfy the GraphWriter contract so the same
    build routine can target Neo4j without any special-casing."""
    assert issubclass(Neo4jGraphBuilder, GraphWriter)


def test_build_kg_into_writes_structure_and_content():
    chunks = [
        Document(
            content="Alpha works at Beta.",
            doc_id="d:chunk0",
            metadata={"parent_doc_id": "d"},
        ),
        Document(
            content="Beta is in Gamma.",
            doc_id="d:chunk1",
            metadata={"parent_doc_id": "d"},
        ),
    ]
    entities = [
        {"id": "alpha", "name": "Alpha", "type": "PERSON"},
        {"id": "beta", "name": "Beta", "type": "ORG"},
    ]
    triples = [("alpha", "works_at", "beta", "Alpha works at Beta.", "d:chunk0")]

    writer = NetworkXGraphWriter()
    stats = build_kg_into(writer, chunks, entities, triples)

    graph = writer.graph

    # 1 document + 2 chunks + 2 entities
    assert stats["nodes_written"] == 5
    assert stats["edges_written"] == 4  # 2×PART_OF + 1×NEXT + 1 semantic

    # Structural nodes and edges
    assert "d" in graph
    assert "d:chunk0" in graph and "d:chunk1" in graph
    assert "PART_OF" in graph.edges["d:chunk0", "d"]["predicates"]
    assert "NEXT" in graph.edges["d:chunk0", "d:chunk1"]["predicates"]

    # Entity and semantic edge
    assert graph.nodes["alpha"]["type"] == "PERSON"
    assert "works_at" in graph.edges["alpha", "beta"]["predicates"]


def test_build_kg_into_stats_reflect_writer_counts():
    writer = NetworkXGraphWriter()
    stats = build_kg_into(writer, [], [], [])
    assert stats == {"nodes_written": 0, "edges_written": 0}
    assert writer.graph.number_of_nodes() == 0
