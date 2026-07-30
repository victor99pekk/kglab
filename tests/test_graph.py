"""Tests for graph construction."""

import networkx as nx
import pytest

from polygraph._shared import Ontology
from polygraph.kg_build.build import GraphBuilder
from polygraph.kg_build.resolve import with_method_and_mapping


def test_build_graph():
    entities = [
        {
            "id": "entity:alice",
            "name": "Alice",
            "type": "PERSON",
            "aliases": ["alice"],
            "confidenceScore": 0.9,
        },
        {
            "id": "entity:acme",
            "name": "Acme Corp",
            "type": "ORG",
            "aliases": ["acme corp"],
            "confidenceScore": 0.95,
        },
    ]
    triples = [("entity:alice", "works_at", "entity:acme")]

    builder = GraphBuilder()
    graph = builder.build(entities, triples)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge("entity:alice", "entity:acme")
    assert graph.nodes["entity:alice"]["name"] == "Alice"
    assert "works_at" in graph.edges["entity:alice", "entity:acme"]["predicates"]


def test_graph_builder_does_not_merge_nodes_with_same_name():
    entities = [
        {"id": "document:one", "name": "article.json", "type": "Document"},
        {"id": "document:two", "name": "article.json", "type": "Document"},
    ]

    graph = GraphBuilder().build(entities, [])

    assert set(graph.nodes) == {"document:one", "document:two"}


def test_graph_edge_preserves_relationship_provenance():
    entities = [
        {"id": "entity:alice", "name": "Alice", "type": "PERSON"},
        {"id": "entity:acme", "name": "Acme", "type": "ORG"},
    ]
    triples = [
        (
            "entity:alice",
            "works_at",
            "entity:acme",
            "Alice works at Acme.",
            "chunk:123",
        )
    ]

    graph = GraphBuilder().build(entities, triples)
    edge = graph.edges["entity:alice", "entity:acme"]

    assert edge["relations"] == [
        {
            "predicate": "works_at",
            "evidence_sentence": "Alice works at Acme.",
            "source_chunk_id": "chunk:123",
        }
    ]


def test_graph_rejects_invalid_structural_relationship_types():
    ontology = Ontology(
        relationship_types={
            "appears_in": {
                "domain": "Entity",
                "range": "Chunk",
                "kind": "structural",
            }
        }
    )
    entities = [
        {"id": "entity:alice", "name": "Alice", "type": "PERSON"},
        {"id": "entity:bob", "name": "Bob", "type": "PERSON"},
    ]

    with pytest.raises(ValueError, match=r"expected Entity->Chunk, got PERSON->PERSON"):
        GraphBuilder(ontology=ontology).build(
            entities,
            [("entity:alice", "appears_in", "entity:bob")],
        )


def test_graph_rejects_structural_relationship_with_missing_endpoint():
    ontology = Ontology(
        relationship_types={
            "part_of": {
                "domain": "Chunk",
                "range": "Document",
                "kind": "structural",
            }
        }
    )
    entities = [{"id": "chunk:one", "name": "Chunk one", "type": "Chunk"}]

    with pytest.raises(ValueError, match=r"missing endpoint.*document:missing"):
        GraphBuilder(ontology=ontology).build(
            entities,
            [("chunk:one", "part_of", "document:missing")],
        )


def test_graph_accepts_valid_structural_relationship_types():
    ontology = Ontology(
        relationship_types={
            "appears_in": {
                "domain": "Entity",
                "range": "Chunk",
                "kind": "structural",
            }
        }
    )
    entities = [
        {"id": "entity:alice", "name": "Alice", "type": "PERSON"},
        {"id": "chunk:one", "name": "Chunk one", "type": "Chunk"},
    ]

    graph = GraphBuilder(ontology=ontology).build(
        entities,
        [("entity:alice", "appears_in", "chunk:one")],
    )

    assert graph.has_edge("entity:alice", "chunk:one")


def test_deduplication_removes_exact_duplicates():
    from polygraph._shared import Document
    from polygraph.preprocess.dedup import Deduplicator

    docs = [
        Document(content="Unique document one.", doc_id="1"),
        Document(content="Unique document two.", doc_id="2"),
        Document(content="Unique document one.", doc_id="3"),  # exact dup of doc 1
    ]

    dedup = Deduplicator(method="ngram", threshold=0.9)
    result = dedup.deduplicate(docs)

    assert len(result) == 2
    ids = {d.doc_id for d in result}
    assert ids == {"1", "2"} or ids == {"2", "3"}


def test_semantic_deduplication_is_selectable_with_multilingual_embeddings():
    from polygraph._shared import Document
    from polygraph.preprocess.dedup import Deduplicator

    documents = [
        Document(content="London is the capital of England.", doc_id="a"),
        Document(content="The capital of England is London.", doc_id="b"),
        Document(content="Tên lửa bay vào không gian.", doc_id="c"),
    ]
    embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    dedup = Deduplicator(
        method="semantic",
        semantic_threshold=0.95,
        semantic_encoder=lambda _texts: embeddings,
    )

    result = dedup.deduplicate(documents)

    assert [document.doc_id for document in result] == ["a", "c"]


def test_embedding_resolution_does_not_merge_semantically_related_names():
    from polygraph.kg_build.resolve import EntityResolver

    entities = [
        {"id": "a", "name": "khoa học", "type": "CONCEPT", "aliases": []},
        {"id": "b", "name": "công nghệ", "type": "CONCEPT", "aliases": []},
    ]
    resolver = EntityResolver(
        method="embedding",
        threshold=0.8,
        encoder=lambda _texts: [[1.0, 0.0], [1.0, 0.0]],
    )

    assert len(resolver.resolve(entities)) == 2


def test_resolution_maps_merged_entity_ids_to_canonical_id():
    entities = [
        {"id": "entity:one", "name": "Alice Smith", "type": "PERSON", "aliases": []},
        {"id": "entity:two", "name": "Alice", "type": "PERSON", "aliases": []},
    ]

    resolved, id_map = with_method_and_mapping(
        entities,
        method="string",
        threshold=0.4,
    )

    assert len(resolved) == 1
    assert id_map == {
        "entity:one": "entity:one",
        "entity:two": "entity:one",
    }


def test_quality_filter_removes_short_docs():
    from polygraph._shared import Document
    from polygraph.preprocess.quality import QualityFilter

    docs = [
        Document(content="Short."),
        Document(
            content="A properly sized document with enough words to pass the quality filter check."
        ),
    ]

    qf = QualityFilter(min_chars=40, min_words=5)
    result = qf.filter(docs)
    assert len(result) == 1
