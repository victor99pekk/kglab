"""Tests for the storage-agnostic graph store (read-side abstraction).

The point: the same ``GraphStore`` interface reads a graph no matter where it
is stored — in-memory ``nx.DiGraph``, SQLite file, JSON export, or Neo4j.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

from kglab._shared.stage_config import BuildConfig, ExtractionConfig, PreprocessConfig
from kglab.kg_build.build import SQLiteGraphWriter
from kglab.kg_eval import evaluate_kg
from kglab.kg_export import exporter
from kglab.kg_export.graph_store import (
    GRAPH_STORE_BACKENDS,
    GraphStore,
    JSONGraphStore,
    Neo4jGraphStore,
    NetworkXGraphStore,
    SQLiteGraphStore,
    create_graph_store,
)
from kglab.pipelines import Baseline


def _sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("e1", id="e1", name="Einstein", type="PERSON")
    graph.add_node("e2", id="e2", name="Princeton", type="ORG")
    graph.add_edge("e1", "e2", predicates=["worked_at"], weight=1)
    return graph


def test_networkx_store_reads_graph():
    store = NetworkXGraphStore(_sample_graph())

    assert store.number_of_nodes() == 2
    assert store.number_of_edges() == 1
    assert store.has_node("e1")
    assert not store.has_node("missing")
    assert store.get_node("e1")["name"] == "Einstein"
    assert store.get_node("missing") is None

    assert store.has_edge("e1", "e2")
    assert not store.has_edge("e2", "e1")
    assert store.get_edge("e1", "e2")["predicates"] == ["worked_at"]
    assert store.get_edge("e2", "e1") is None

    assert list(store.successors("e1")) == ["e2"]
    assert list(store.predecessors("e2")) == ["e1"]
    assert store.out_degree("e1") == 1
    assert store.in_degree("e2") == 1

    assert [m["id"] for m in store.search("stein")] == ["e1"]
    assert store.stats == {"num_nodes": 2, "num_edges": 1}

    store.close()  # no-op for in-memory


def test_networkx_store_materialises_same_graph():
    original = _sample_graph()
    store = NetworkXGraphStore(original)
    assert store.to_networkx() is original  # already in-memory, no copy


def test_networkx_store_is_graph_store():
    assert issubclass(NetworkXGraphStore, GraphStore)


def test_json_store_round_trips_export(tmp_path):
    exporter.to_json(_sample_graph(), [], [], tmp_path / "knowledge_graph.json")

    store = create_graph_store("json", path=tmp_path / "knowledge_graph.json")

    assert store.number_of_nodes() == 2
    assert store.number_of_edges() == 1
    assert store.get_node("e1")["name"] == "Einstein"
    assert store.has_edge("e1", "e2")
    assert [m["id"] for m in store.search("Princeton")] == ["e2"]

    # Rebuilt via to_networkx must match counts too
    rebuilt = store.to_networkx()
    assert rebuilt.number_of_nodes() == 2
    assert rebuilt.number_of_edges() == 1

    store.close()


def test_sqlite_store_reads_file_backed_graph(tmp_path):
    writer = SQLiteGraphWriter(db_path=str(tmp_path / "kg.db"))
    writer.merge_entity("e1", name="Einstein", entity_type="PERSON")
    writer.merge_edge(
        "e1",
        "e2",
        "worked_at",
        evidence_sentence="evidence one",
        source_chunk_id="chunk1",
    )
    graph = writer.graph

    store = create_graph_store("sqlite", graph=graph)

    assert store.number_of_nodes() == 2
    assert store.number_of_edges() == 1
    assert store.get_node("e1")["name"] == "Einstein"
    assert list(store.successors("e1")) == ["e2"]
    assert list(store.predecessors("e2")) == ["e1"]
    assert store.out_degree("e1") == 1
    assert store.in_degree("e2") == 1
    assert store.search("Einstein")

    store.close()


def test_create_graph_store_unknown_backend():
    with pytest.raises(ValueError, match="Unknown graph store backend"):
        create_graph_store("parquet")


def test_backend_registry_has_all_formats():
    for backend in ("networkx", "sqlite", "neo4j", "json", "graphml"):
        assert backend in GRAPH_STORE_BACKENDS


# ── Pipeline integration: graph_store returned from build/execute ──


def _make_corpus(tmp_path: Path, texts: list[str]) -> Path:
    """Write a small JSONL corpus (id + text) for real preprocessing."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i, text in enumerate(texts):
            fh.write(json.dumps({"id": f"doc{i}", "text": text}) + "\n")
    return path


_ALICE = "Alice founded Acme Corp in 1999. She was the first female CEO and led the company through a major expansion across Europe and Asia."
_BOB = "Bob worked at Acme Corp since 2010. He joined as a senior engineer and later became the head of research in the London office."


def _baseline(tmp_path: Path, **kwargs) -> Baseline:
    pipe = Baseline(
        preprocess=PreprocessConfig(
            quality_min_chars=50,
            quality_min_words=10,
            chunk_method="sentence",
            doc_dedup_method="minhash",
            chunk_dedup_method="minhash",
        ),
        extraction=ExtractionConfig(
            mode="composed", entity_method="regex", relation_method="ontology_rules"
        ),
        **kwargs,
    )
    pipe.output_dir = Path(tmp_path)  # constructor kwargs are not applied to attrs
    return pipe


def _build_kg(pipe: Baseline, corpus: Path) -> dict:
    pipe.input_paths = [corpus]
    return pipe.build_kg(pipe.preprocess())


def test_baseline_build_kg_returns_in_memory_graph_store(tmp_path):
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path)
    kg = _build_kg(pipe, corpus)

    store = kg["graph_store"]
    assert isinstance(store, NetworkXGraphStore)
    assert pipe.graph_store is store
    assert store.number_of_nodes() == kg["graph"].number_of_nodes()
    assert store.number_of_edges() == kg["graph"].number_of_edges()
    assert store.search("Alice")  # interactive querying works
    store.close()


def test_baseline_sqlite_build_returns_sqlite_graph_store(tmp_path):
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path, build=BuildConfig(method="sqlite"))
    kg = _build_kg(pipe, corpus)

    store = kg["graph_store"]
    assert isinstance(store, SQLiteGraphStore)
    assert store.number_of_nodes() >= 1
    store.close()


def test_baseline_continues_building_on_existing_store(tmp_path):
    # First run: build a KG and keep its store.
    corpus1 = _make_corpus(tmp_path / "run1", [_ALICE])
    pipe1 = _baseline(tmp_path / "run1")
    kg1 = _build_kg(pipe1, corpus1)
    store1 = kg1["graph_store"]
    alice = store1.search("Alice")
    assert alice
    alice_id = alice[0]["id"]

    # Second run: continue building on top of the existing graph.
    corpus2 = _make_corpus(tmp_path / "run2", [_BOB])
    pipe2 = _baseline(tmp_path / "run2")
    pipe2.input_paths = [corpus2]
    kg2 = pipe2.build_kg(pipe2.preprocess(), existing_store=store1)
    store2 = kg2["graph_store"]

    assert store2.number_of_nodes() >= store1.number_of_nodes()
    assert store2.has_node(alice_id)  # prior entities preserved
    assert store2.search("Bob")  # new entities added
    store1.close()
    store2.close()


def test_evaluate_kg_accepts_graph_store(tmp_path):
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path)
    kg = _build_kg(pipe, corpus)

    # Evaluate from the store alone (no nx graph in the dict).
    report = evaluate_kg(
        {
            "graph_store": kg["graph_store"],
            "entities": kg["entities"],
            "triples": kg["triples"],
        }
    )
    assert report["num_nodes"] == kg["graph"].number_of_nodes()
    assert report["num_edges"] == kg["graph"].number_of_edges()


# ── Selectable graph_store backend ──────────────────────────────


def test_baseline_graph_store_backend_networkx_explicit(tmp_path):
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path, graph_store_backend="networkx")
    kg = _build_kg(pipe, corpus)
    assert isinstance(kg["graph_store"], NetworkXGraphStore)


def test_baseline_graph_store_backend_json(tmp_path):
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path, graph_store_backend="json")
    kg = _build_kg(pipe, corpus)

    store = kg["graph_store"]
    assert isinstance(store, JSONGraphStore)
    assert (pipe.output_dir / "knowledge_graph.json").exists()
    assert store.number_of_nodes() == kg["graph"].number_of_nodes()
    store.close()


class _FakeResult:
    def __init__(self, value=None):
        self._value = value

    def single(self):
        return [self._value] if self._value is not None else None

    def __iter__(self):
        return iter(())

    def consume(self):
        return None


class _FakeSession:
    def __init__(self):
        self.queries: list[str] = []

    def run(self, query, **params):
        self.queries.append(query)
        return _FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeDriver:
    def __init__(self):
        self.session_ = _FakeSession()

    def session(self):
        return self.session_

    def close(self):
        pass


def test_baseline_graph_store_backend_neo4j_streams(tmp_path, monkeypatch):
    """build_kg streams directly into Neo4j — no in-memory graph at all."""
    from kglab.kg_export.neo4j import upload as neo4j_upload

    driver = _FakeDriver()
    monkeypatch.setattr(neo4j_upload, "_get_connection", lambda **kw: driver)

    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(
        tmp_path,
        graph_store_backend="neo4j",
        graph_store_options={"clear": True, "uri": "bolt://example:7687"},
    )
    kg = _build_kg(pipe, corpus)

    store = kg["graph_store"]
    assert isinstance(store, Neo4jGraphStore)
    assert kg["graph"] is None  # the graph lives in Neo4j, not RAM
    assert kg["neo4j_stats"]["nodes_written"] > 0
    assert kg["neo4j_stats"]["edges_written"] > 0

    # build_kg_into ran real MERGE statements against the fake session
    joined = "\n".join(driver.session_.queries)
    assert "CREATE CONSTRAINT" in joined  # schema ensured
    assert "MERGE (c:Chunk" in joined  # chunks streamed
    assert "MERGE (d:Document" in joined  # documents streamed
    assert "MERGE (n:Entity" in joined  # entities streamed
    assert "PART_OF" in joined  # structural edges
    assert "APPEARS_IN" in joined  # entity→chunk edges

    store.close()


def test_baseline_accepts_configured_neo4j_store(tmp_path, monkeypatch):
    """Credentials live on the store — pass the store, not messy args."""
    from kglab.kg_export.neo4j import upload as neo4j_upload

    connection_calls: list[dict] = []

    def fake_get_connection(**kwargs):
        connection_calls.append(kwargs)
        return _FakeDriver()

    monkeypatch.setattr(neo4j_upload, "_get_connection", fake_get_connection)

    # Set credentials once on the store…
    store = Neo4jGraphStore(uri="bolt://creds:7687", user="neo", password="secret")

    # …then pass the store itself into the pipeline (no graph_store_options).
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path)
    pipe.input_paths = [corpus]
    kg = pipe.build_kg(pipe.preprocess(), graph_store=store)

    # The same store instance is returned, and it supplied the credentials.
    assert kg["graph_store"] is store
    assert kg["graph"] is None
    assert connection_calls[-1] == {
        "uri": "bolt://creds:7687",
        "user": "neo",
        "password": "secret",
    }

    store.close()


def test_baseline_execute_neo4j_streams_end_to_end(tmp_path, monkeypatch):
    """Full execute() works with the streaming backend (export skips gracefully)."""
    from kglab.kg_export.neo4j import upload as neo4j_upload

    driver = _FakeDriver()
    monkeypatch.setattr(neo4j_upload, "_get_connection", lambda **kw: driver)

    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path, graph_store_backend="neo4j")
    kg = pipe.execute(
        input_paths=[corpus],
        output_dir=tmp_path / "out",
        graph_store_options={"clear": True},
    )

    assert kg["graph"] is None
    assert isinstance(kg["graph_store"], Neo4jGraphStore)
    assert kg["neo4j_stats"]["nodes_written"] > 0

    # External evaluation (as main.py does) writes stats-based metrics.
    report = evaluate_kg(kg, output_dir=tmp_path / "out")
    assert (tmp_path / "out" / "metrics.json").exists()
    assert report["num_nodes"] > 0
    kg["graph_store"].close()


def test_baseline_graph_store_backend_unknown_raises(tmp_path):
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path, graph_store_backend="parquet")
    with pytest.raises(ValueError, match="Unknown graph_store_backend"):
        _build_kg(pipe, corpus)


def test_baseline_unified_build_preserves_entity_attributes(tmp_path):
    """The unified build_kg_into path keeps full entity attrs in the graph."""
    corpus = _make_corpus(tmp_path, [_ALICE])
    pipe = _baseline(tmp_path)
    kg = _build_kg(pipe, corpus)

    nodes_with_provenance = [
        node
        for node, data in kg["graph"].nodes(data=True)
        if data.get("type") not in ("Chunk", "Document") and data.get("source_chunk_ids")
    ]
    assert nodes_with_provenance, (
        "entity nodes should retain source_chunk_ids through the unified build"
    )
