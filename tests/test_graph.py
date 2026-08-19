"""Tests for graph construction."""

import networkx as nx
import pytest

from kglab._shared import Ontology
from kglab.kg_build.build import GraphBuilder
from kglab.kg_build.resolve import with_method_and_mapping
from kglab.pipelines.baseline import _entity_chunk_membership_triples


def test_build_graph():
    entities = [
        {
            "id": "entity:alice",
            "name": "Alice",
            "type": "PERSON",
            "aliases": ["alice"],
            "confidenceScore": 0.9,
            "source_chunk_ids": ["chunk:one"],
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
    assert graph.nodes["entity:alice"]["source_chunk_ids"] == ["chunk:one"]
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


def test_graph_validation_accepts_structural_nodes_and_case_insensitive(caplog):
    """Document/Chunk structural nodes and case variants never warn.

    Every KG contains document and chunk nodes even though those are not
    *entity* types in the ontology, and matching is case-insensitive
    (``Document`` vs an ontology-declared ``DOCUMENT``).
    """
    ontology = Ontology(
        entity_types={
            "PERSON": {"description": "a person"},
            "DOCUMENT": {"description": "a document"},
        }
    )
    entities = [
        {"id": "entity:alice", "name": "Alice", "type": "PERSON"},
        {"id": "document:one", "name": "doc one", "type": "Document"},
        {"id": "chunk:one", "name": "chunk one", "type": "Chunk"},
    ]

    with caplog.at_level("WARNING", logger="kglab.kg_build.build.networkx"):
        GraphBuilder(ontology=ontology).build(entities, [])

    assert "not in schema" not in caplog.text


def test_graph_validation_warns_for_undeclared_entity_type(caplog):
    """A genuinely undeclared semantic entity type still warns."""
    ontology = Ontology(
        entity_types={"PERSON": {"description": "a person"}},
    )
    entities = [
        {"id": "entity:alice", "name": "Alice", "type": "PERSON"},
        {"id": "entity:two", "name": "two", "type": "CARDINAL"},
    ]

    with caplog.at_level("WARNING", logger="kglab.kg_build.build.networkx"):
        GraphBuilder(ontology=ontology).build(entities, [])

    assert "node labels not in schema" in caplog.text


def test_deduplication_removes_exact_duplicates():
    from kglab._shared import Document
    from kglab.preprocess.dedup import Deduplicator

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
    from kglab._shared import Document
    from kglab.preprocess.dedup import Deduplicator

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
    from kglab.kg_build.resolve import EntityResolver

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
        {
            "id": "entity:one",
            "name": "Alice Smith",
            "type": "PERSON",
            "aliases": [],
            "source_chunk_ids": ["chunk:one"],
        },
        {
            "id": "entity:two",
            "name": "Alice",
            "type": "PERSON",
            "aliases": [],
            "source_chunk_ids": ["chunk:two"],
        },
    ]

    resolved, id_map = with_method_and_mapping(
        entities,
        method="string",
        threshold=0.4,
    )

    assert len(resolved) == 1
    assert resolved[0]["source_chunk_ids"] == ["chunk:one", "chunk:two"]
    assert id_map == {
        "entity:one": "entity:one",
        "entity:two": "entity:one",
    }


def test_embedding_resolution_merges_source_chunk_provenance():
    entities = [
        {
            "id": "entity:one",
            "name": "Alice Smith",
            "type": "PERSON",
            "aliases": [],
            "source_chunk_ids": ["chunk:one"],
        },
        {
            "id": "entity:two",
            "name": "Alice",
            "type": "PERSON",
            "aliases": [],
            "source_chunk_ids": ["chunk:two"],
        },
    ]

    resolved, _ = with_method_and_mapping(
        entities,
        method="embedding",
        threshold=0.8,
        encoder=lambda _texts: [[1.0, 0.0], [1.0, 0.0]],
    )

    assert len(resolved) == 1
    assert resolved[0]["source_chunk_ids"] == ["chunk:one", "chunk:two"]


def test_relationless_entity_still_gets_chunk_membership():
    entities = [
        {
            "id": "entity:alice",
            "name": "Alice",
            "type": "PERSON",
            "source_chunk_ids": ["chunk:one"],
        }
    ]

    triples = _entity_chunk_membership_triples(entities, {"chunk:one"})

    assert triples == [("entity:alice", "appears_in", "chunk:one", "", "chunk:one")]


def test_quality_filter_removes_short_docs():
    from kglab._shared import Document
    from kglab.preprocess.quality import QualityFilter

    docs = [
        Document(content="Short."),
        Document(
            content="A properly sized document with enough words to pass the quality filter check."
        ),
    ]

    qf = QualityFilter(min_chars=40, min_words=5)
    result = qf.filter(docs)
    assert len(result) == 1


def test_quality_filter_uses_new_default_thresholds():
    """Documents below 200 chars / 40 words are rejected at default config."""
    from kglab._shared import Document
    from kglab._shared.stage_config import PreprocessConfig
    from kglab.preprocess.quality import QualityFilter

    cfg = PreprocessConfig()
    assert cfg.quality_min_chars == 200
    assert cfg.quality_min_words == 40

    # A document below the new thresholds should be filtered
    short_doc = Document(
        content="Short text with only a handful of words.",
        doc_id="short",
    )

    qf = QualityFilter(
        min_chars=cfg.quality_min_chars,
        min_words=cfg.quality_min_words,
    )
    result = qf.filter([short_doc])
    assert len(result) == 0


def test_layered_dedup_combines_minhash_and_semantic():
    """Layered dedup method chains MinHash → Semantic deduplication."""
    from kglab._shared import Document
    from kglab.preprocess.dedup import Deduplicator

    docs = [
        Document(content="Marie Curie discovered radium in 1898.", doc_id="a"),
        Document(content="Radium was discovered by Marie Curie in 1898.", doc_id="b"),
        Document(content="The Eiffel Tower is in Paris, France.", doc_id="c"),
    ]

    # Use fake encoder so we don't download a model
    dedup = Deduplicator(
        method="layered",
        threshold=0.85,
        semantic_threshold=0.95,
        semantic_encoder=lambda _texts: [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]],
    )
    result = dedup.deduplicate(docs)

    # "a" and "b" are semantic near-duplicates → one removed
    # "c" is different → kept
    kept_ids = {d.doc_id for d in result}
    assert "c" in kept_ids
    assert len(result) == 2


def test_layered_dedup_raises_when_too_many_records():
    """When records exceed max, layered dedup raises instead of silently skipping.

    The library fails loudly: the semantic layer is never silently dropped in
    favor of MinHash-only results.
    """
    from kglab._shared import Document
    from kglab.preprocess.dedup import Deduplicator

    # Create more documents than semantic_max_records (5000)
    docs = [
        Document(content=f"Document number {i} with unique content.", doc_id=str(i))
        for i in range(10)
    ]

    dedup = Deduplicator(
        method="layered",
        threshold=0.85,
        semantic_max_records=5,  # artificially low
    )

    with pytest.raises(ValueError, match=r"Semantic dedup cannot run on 10 records"):
        dedup.deduplicate(docs)
