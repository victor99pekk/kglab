"""Tests for canonical KG upload into Neo4j without a live database."""

import json

import pytest

from polygraph.kg_export.graph_db.neo4j import Neo4jUploader
from polygraph.kg_export.neo4j import upload
from polygraph.kg_export.neo4j.builder import Neo4jGraphBuilder, _safe_rel_type


class _Result:
    def __init__(self, record=None):
        self._record = record

    def single(self):
        return self._record

    def consume(self):
        return self

    def __iter__(self):
        return iter(())


class _FakeSession:
    def __init__(self, replacement_record=None):
        self.calls = []
        self.replacement_record = replacement_record or {
            "chunk_ids": [],
            "entity_ids": [],
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        if "RETURN collect(DISTINCT chunk.id) AS chunk_ids" in query:
            return _Result(self.replacement_record)
        return _Result()


class _FakeDriver:
    def __init__(self, session):
        self._session = session
        self.closed = False

    def session(self):
        return self._session

    def close(self):
        self.closed = True


def _queries(session):
    return [query for query, _ in session.calls]


def _matching_calls(session, pattern):
    return [(query, parameters) for query, parameters in session.calls if pattern in query]


def test_upload_prefers_entities_and_triples_and_needs_no_apoc(tmp_path, monkeypatch):
    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "graph": {
                    "nodes": [
                        {
                            "id": "entity:alice",
                            "name": "Alice",
                            "type": "PERSON",
                            "importanceScore": 0.75,
                        },
                        {
                            "id": "entity:acme",
                            "name": "Acme",
                            "type": "ORG",
                        },
                        {
                            "id": "chunk:one",
                            "type": "Chunk",
                            "text": "Alice works at Acme.",
                        },
                        {
                            "id": "document:one",
                            "name": "Article",
                            "type": "Document",
                        },
                    ],
                    "edges": [
                        {
                            "source": "entity:alice",
                            "target": "document:one",
                            "predicates": ["legacy_should_not_upload"],
                        }
                    ],
                },
                "entities": [
                    {
                        "id": "entity:alice",
                        "name": "Alice",
                        "type": "PERSON",
                        "aliases": ["A. Example"],
                        "source_chunk_ids": ["chunk:one"],
                    },
                    {
                        "id": "entity:acme",
                        "name": "Acme",
                        "type": "ORG",
                    },
                    {
                        "id": "chunk:one",
                        "type": "Chunk",
                        "text": "Alice works at Acme.",
                    },
                    {
                        "id": "document:one",
                        "name": "Article",
                        "type": "Document",
                    },
                ],
                "triples": [
                    {
                        "subject": "entity:alice",
                        "predicate": "appears_in",
                        "object": "chunk:one",
                        "source_chunk_id": "chunk:one",
                    },
                    {
                        "subject": "chunk:one",
                        "predicate": "part_of",
                        "object": "document:one",
                        "source_chunk_id": "chunk:one",
                    },
                    {
                        "subject": "entity:alice",
                        "predicate": "works_at",
                        "object": "entity:acme",
                        "evidence_sentence": "Alice works at Acme.",
                        "source_chunk_id": "chunk:one",
                    },
                    {
                        "subject": "entity:alice",
                        "predicate": "works_at",
                        "object": "entity:acme",
                        "evidence_sentence": "Alice joined Acme.",
                        "source_chunk_id": "chunk:two",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    session = _FakeSession()
    driver = _FakeDriver(session)
    monkeypatch.setattr(upload, "_get_connection", lambda **_kwargs: driver)

    upload.upload_graph(graph_path, clear=True)

    all_queries = "\n".join(_queries(session))
    assert "apoc." not in all_queries
    assert "LEGACY_SHOULD_NOT_UPLOAD" not in all_queries
    assert "MERGE (a)-[rel:APPEARS_IN]->(b)" in all_queries
    assert "MERGE (a)-[rel:PART_OF]->(b)" in all_queries
    assert "MERGE (a)-[rel:WORKS_AT]->(b)" in all_queries
    assert all_queries.count("CREATE CONSTRAINT") == 3
    assert driver.closed

    entity_upload = _matching_calls(session, "MERGE (n:Entity {id: item.id})")
    entity_rows = entity_upload[0][1]["batch"]
    alice = next(row for row in entity_rows if row["id"] == "entity:alice")
    assert alice["type"] == "PERSON"
    assert alice["aliases"] == ["A. Example"]
    assert alice["importance_score"] == 0.75
    assert "source_chunk_ids" not in alice

    semantic_upload = _matching_calls(session, "MERGE (a)-[rel:WORKS_AT]->(b)")
    assert semantic_upload[0][1]["rows"] == [
        {
            "source": "entity:alice",
            "target": "entity:acme",
            "evidence_sentences": [
                "Alice joined Acme.",
                "Alice works at Acme.",
            ],
            "source_chunk_ids": ["chunk:one", "chunk:two"],
            "description": "",
        }
    ]

    structural_upload = _matching_calls(session, "MERGE (a)-[rel:APPEARS_IN]->(b)")
    assert "sourceChunkIds" not in structural_upload[0][0]


def test_public_neo4j_uploader_forwards_connection_options(tmp_path, monkeypatch):
    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "graph": {
                    "nodes": [
                        {
                            "id": "entity:alice",
                            "name": "Alice",
                            "type": "PERSON",
                        }
                    ],
                    "edges": [],
                },
                "entities": [
                    {
                        "id": "entity:alice",
                        "name": "Alice",
                        "type": "PERSON",
                    }
                ],
                "triples": [],
            }
        ),
        encoding="utf-8",
    )
    session = _FakeSession()
    driver = _FakeDriver(session)
    connection_options = {}

    def fake_connection(**kwargs):
        connection_options.update(kwargs)
        return driver

    monkeypatch.setattr(upload, "_get_connection", fake_connection)

    Neo4jUploader(
        uri="bolt://neo4j.example:7687",
        user="graph-user",
        password="secret",
    ).upload(graph_path, clear=True)

    assert connection_options == {
        "uri": "bolt://neo4j.example:7687",
        "user": "graph-user",
        "password": "secret",
    }
    assert "MATCH (n) DETACH DELETE n" in "\n".join(_queries(session))
    assert driver.closed


def test_connection_overrides_reach_neo4j_driver(monkeypatch):
    from neo4j import GraphDatabase

    captured = {}
    expected_driver = object()

    def fake_driver(uri, *, auth):
        captured.update({"uri": uri, "auth": auth})
        return expected_driver

    monkeypatch.setattr(GraphDatabase, "driver", fake_driver)

    driver = upload._get_connection(
        uri="bolt://neo4j.example:7687",
        user="graph-user",
        password="secret",
    )

    assert driver is expected_driver
    assert captured == {
        "uri": "bolt://neo4j.example:7687",
        "auth": ("graph-user", "secret"),
    }


def test_legacy_graph_edges_remain_uploadable():
    data = {
        "graph": {
            "nodes": [
                {"id": "entity:alice", "name": "Alice", "type": "PERSON"},
                {"id": "entity:bob", "name": "Bob", "type": "PERSON"},
            ],
            "edges": [
                {
                    "source": "entity:alice",
                    "target": "entity:bob",
                    "predicates": ["knows"],
                    "relations": [
                        {
                            "predicate": "knows",
                            "evidence_sentence": "Alice knows Bob.",
                            "source_chunk_id": "chunk:one",
                        }
                    ],
                }
            ],
        }
    }

    assert [node["id"] for node in upload._canonical_nodes(data)] == [
        "entity:alice",
        "entity:bob",
    ]
    assert upload._canonical_relationships(data) == [
        {
            "source": "entity:alice",
            "predicate": "knows",
            "target": "entity:bob",
            "evidence_sentence": "Alice knows Bob.",
            "source_chunk_id": "chunk:one",
            "description": "",
        }
    ]


def test_upload_rejects_relationship_with_missing_endpoint():
    with pytest.raises(ValueError, match="entity:missing"):
        upload._validate_endpoints(
            [{"id": "entity:alice", "type": "PERSON"}],
            [
                {
                    "source": "entity:alice",
                    "predicate": "knows",
                    "target": "entity:missing",
                }
            ],
        )


def test_upload_rejects_invalid_structural_endpoint_types():
    nodes = [
        {"id": "entity:alice", "type": "PERSON"},
        {"id": "entity:bob", "type": "PERSON"},
    ]

    with pytest.raises(ValueError, match=r"expected Entity->Chunk, got PERSON->PERSON"):
        upload._validate_endpoints(
            nodes,
            [
                {
                    "source": "entity:alice",
                    "predicate": "appears_in",
                    "target": "entity:bob",
                }
            ],
        )


def test_document_replacement_uses_appears_in_and_preserves_other_provenance():
    session = _FakeSession(
        {
            "chunk_ids": ["chunk:old"],
            "entity_ids": ["entity:alice"],
        }
    )

    upload.replace_documents(session, ["document:one"])

    all_queries = "\n".join(_queries(session))
    assert "OPTIONAL MATCH (entity:Entity)-[:APPEARS_IN]->(chunk)" in all_queries
    assert "NOT type(relationship) IN ['APPEARS_IN', 'PART_OF', 'NEXT']" in all_queries
    assert "SET relationship.sourceChunkIds = remaining_chunk_ids" in all_queries
    assert "relationship.evidenceSentences = []" not in all_queries
    assert "NOT EXISTS { MATCH (entity)-[:APPEARS_IN]->(:Chunk) }" in all_queries
    assert "MENTIONS" not in all_queries


def test_streaming_builder_uses_same_schema_and_structural_relationships():
    session = _FakeSession()
    builder = Neo4jGraphBuilder(session)

    builder.merge_structural_edge("entity:alice", "chunk:one", "appears_in")
    builder.clear_database()

    all_queries = "\n".join(_queries(session))
    assert all_queries.count("CREATE CONSTRAINT") == 6
    assert "MERGE (a)-[:APPEARS_IN]->(b)" in all_queries
    assert "DROP CONSTRAINT" not in all_queries
    assert "DROP INDEX" not in all_queries


def test_relationship_type_is_safe_for_generated_cypher():
    assert _safe_rel_type("works at / for") == "WORKS_AT___FOR"
    assert _safe_rel_type("123") == "REL_123"
    assert _safe_rel_type("---") == "RELATION"
