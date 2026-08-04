"""Abstract streaming pipeline — builds directly into Neo4j to avoid OOM.

Unlike ``Pipeline``, which accumulates the full knowledge graph as an
``nx.DiGraph`` in RAM, this class streams entities, chunks, and triples
directly into a Neo4j database via ``Neo4jGraphBuilder``.  This makes it
possible to build graphs larger than available memory.

Subclass and implement ``preprocess()`` and ``build_kg_streaming()``.

Usage::

    from polygraph.pipelines.streaming import StreamingPipeline
    from polygraph.kg_export.neo4j.builder import Neo4jGraphBuilder

    class MyStreamingPipeline(StreamingPipeline):
        def preprocess(self) -> list[Document]: ...
        def build_kg_streaming(self, chunks, builder: Neo4jGraphBuilder) -> None: ...

    pipe = MyStreamingPipeline(
        input_paths=["data/"],
        output_dir="output/",
    )
    pipe.execute()
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

from polygraph._shared import Document
from polygraph.kg_export.neo4j.builder import Neo4jGraphBuilder
from polygraph.pipelines._base import Pipeline


def _export_graphml_from_neo4j(session: Any, path: Path) -> None:
    """Build an in-memory ``nx.DiGraph`` from Neo4j and write GraphML.

    .. warning::
       This materialises the full graph in RAM.  Prefer JSON export
       (which streams) for large graphs, or use Neo4j's native
       ``neo4j-admin`` dump tools.
    """
    import networkx as nx

    from polygraph.kg_export import exporter

    g = nx.DiGraph()

    for record in session.run("MATCH (n) RETURN properties(n) AS props"):
        props = dict(record["props"])
        node_id = props.pop("id", "")
        g.add_node(node_id, **props)

    for record in session.run(
        """
        MATCH (a)-[r]->(b)
        RETURN properties(r) AS props, a.id AS source, b.id AS target, type(r) AS rel_type
        """
    ):
        edge = dict(record["props"])
        edge["type"] = record["rel_type"]
        g.add_edge(
            record["source"],
            record["target"],
            **edge,
        )

    exporter.to_graphml(g, path)
    print(f"[export] GraphML → {path} ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")


class StreamingPipeline(Pipeline):
    """Pipeline that streams nodes and edges directly into a Neo4j database.

    Extends ``Pipeline`` but overrides ``build_kg``, ``evaluate``, ``export``,
    and ``execute`` so that the graph is never fully materialised in RAM.
    Instead, entities and triples are written to Neo4j as they are produced
    via ``Neo4jGraphBuilder``.

    Subclasses must implement:

    * ``preprocess()`` — same as ``Pipeline``, returns cleaned chunks.
    * ``build_kg_streaming(chunks, builder)`` — write nodes and edges to the
      provided ``Neo4jGraphBuilder`` instance.

    Connection parameters default to ``NEO4J_URI`` / ``NEO4J_USER`` /
    ``NEO4J_PASSWORD`` environment variables and can be overridden in the
    constructor.

    Args:
        input_paths: Data files or directories to load.
        output_dir: Where results (metrics, exported JSON) are written.
        uri: Neo4j bolt URI (default: ``$NEO4J_URI`` or ``bolt://localhost:7687``).
        user: Neo4j username (default: ``$NEO4J_USER`` or ``neo4j``).
        password: Neo4j password (default: ``$NEO4J_PASSWORD``).
        clear_db: If ``True``, wipe the database before building.
        **kwargs: Forwarded to ``Pipeline.__init__`` (stored in ``_config``).

    Example::

        class MyStreamingPipeline(StreamingPipeline):
            def preprocess(self) -> list[Document]:
                docs = load.from_paths(self.input_paths)
                return chunk.by_sentence(docs)

            def build_kg_streaming(self, chunks, builder):
                for i, c in enumerate(chunks):
                    builder.merge_chunk(c.doc_id, text=c.content, index=i)

                entities, triples = extract.with_methods(chunks, ontology)
                for e in entities:
                    builder.merge_entity(e["id"], name=e["name"], entity_type=e["type"])
                for s, p, o in triples:
                    builder.merge_edge(s, o, p)
    """

    def __init__(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        clear_db: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_paths, output_dir, **kwargs)

        self._neo4j_uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._neo4j_user = user or os.environ.get("NEO4J_USER", "neo4j")
        self._neo4j_password = password or os.environ.get("NEO4J_PASSWORD", "")
        self._clear_db = clear_db

        self._driver: Any = None
        self._session: Any = None
        self._builder: Neo4jGraphBuilder | None = None

    # ── Abstract streaming stage ───────────────────────────────

    @abstractmethod
    def build_kg_streaming(self, chunks: list[Document], builder: Neo4jGraphBuilder) -> None:
        """Stream chunks, entities, and triples directly into Neo4j.

        Use the provided ``builder`` to write nodes and edges.  Typical
        implementation pattern::

            def build_kg_streaming(self, chunks, builder):
                # 1. Write chunk and document nodes
                for i, chunk in enumerate(chunks):
                    builder.merge_chunk(chunk.doc_id, text=chunk.content, index=i)

                # 2. Extract entities and triples (still returned, but small batches)
                entities, triples = extract.with_methods(chunks, ontology, ...)

                # 3. Stream entities to Neo4j
                for entity in entities:
                    builder.merge_entity(
                        entity["id"],
                        name=entity.get("name", ""),
                        entity_type=entity.get("type", "Entity"),
                        description=entity.get("description", ""),
                    )

                # 4. Stream triples to Neo4j
                for subj, pred, obj in triples:
                    builder.merge_edge(subj, obj, pred)

                # 5. Structural edges (chunk→document, chunk→entity, etc.)
                for chunk in chunks:
                    builder.merge_structural_edge(chunk.doc_id, doc_id, "PART_OF")
        """
        ...

    # ── Connection management ──────────────────────────────────

    def _connect(self) -> None:
        """Establish Neo4j connection and create the graph builder."""
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            self._neo4j_uri,
            auth=(self._neo4j_user, self._neo4j_password),
        )
        self._session = self._driver.session()
        self._builder = Neo4jGraphBuilder(self._session)

    def _disconnect(self) -> None:
        """Close Neo4j session and driver."""
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        self._builder = None

    def _require_builder(self) -> Neo4jGraphBuilder:
        """Return the builder, raising if not connected."""
        if self._builder is None:
            raise RuntimeError(
                "Not connected to Neo4j. The StreamingPipeline manages the "
                "connection automatically during execute()."
            )
        return self._builder

    # ── Override Pipeline stages ───────────────────────────────

    def build_kg(self, chunks: list[Document]) -> dict[str, Any]:
        """Stream nodes and edges into Neo4j (no in-memory graph).

        Delegates to ``build_kg_streaming`` and returns a lightweight
        summary dict.  The real graph lives in Neo4j.
        """
        builder = self._require_builder()
        self.build_kg_streaming(chunks, builder)

        stats = builder.stats
        print(
            f"[build_kg] streamed {stats['nodes_written']} nodes, "
            f"{stats['edges_written']} edges to Neo4j"
        )
        return {"graph": None, "entities": [], "triples": [], "neo4j_stats": stats}

    def evaluate(self, kg: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: ARG002
        """Evaluate KG quality by querying Neo4j.

        Computes node/edge counts and label distributions directly from the
        database.  Override to add custom metrics or LLM-based evaluation.
        """
        builder = self._require_builder()
        stats = builder.compute_stats()
        report: dict[str, Any] = {
            "num_nodes": stats.get("num_nodes", 0),
            "num_edges": stats.get("num_edges", 0),
            "label_distribution": stats.get("label_distribution", {}),
        }

        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"[evaluate] {report['num_nodes']} nodes, {report['num_edges']} edges → {path}")
        return report

    def export(self, kg: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        """Export the KG from Neo4j to JSON (and optionally GraphML).

        Streams nodes and edges directly from the Neo4j cursor to disk so
        the full graph is never materialised in RAM.  GraphML export (opt-in
        via ``graphml=True``) still requires building an in-memory
        ``nx.DiGraph`` and is best avoided for very large graphs.
        """
        if self._session is None:
            raise RuntimeError("Not connected to Neo4j.")

        json_path = self.output_dir / "knowledge_graph.json"

        node_count = 0
        edge_count = 0

        with open(json_path, "w") as fh:
            fh.write('{"graph":{"nodes":[')

            # ── Stream nodes ───────────────────────────────────
            first = True
            for record in self._session.run("MATCH (n) RETURN properties(n) AS props"):
                props = dict(record["props"])
                if not first:
                    fh.write(",")
                json.dump(props, fh, default=str)
                first = False
                node_count += 1

            fh.write('],"edges":[')

            # ── Stream edges ───────────────────────────────────
            first = True
            for record in self._session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN properties(r) AS props, a.id AS source, b.id AS target, type(r) AS rel_type
                """
            ):
                edge = dict(record["props"])
                edge["source"] = record["source"]
                edge["target"] = record["target"]
                edge["type"] = record["rel_type"]
                if not first:
                    fh.write(",")
                json.dump(edge, fh, default=str)
                first = False
                edge_count += 1

            fh.write("]}}")

        print(f"[export] {node_count} nodes, {edge_count} edges → {json_path}")

        if self._config.get("graphml"):
            _export_graphml_from_neo4j(
                self._session,
                self.output_dir / "knowledge_graph.graphml",
            )

    # ── Orchestration ──────────────────────────────────────────

    def execute(self) -> None:
        """Full pipeline: connect → clear → preprocess → stream → evaluate → export.

        Manages the Neo4j connection lifecycle so subclasses only need to
        implement ``preprocess()`` and ``build_kg_streaming()``.
        """
        print(f"=== {self.__class__.__name__} ===")
        print(f"Input:  {self.input_paths}")
        print(f"Output: {self.output_dir}")
        print(f"Neo4j:  {self._neo4j_uri}\n")

        self._connect()
        try:
            if self._clear_db:
                print("[neo4j] Clearing database...")
                self._require_builder().clear_database()

            super().execute()
        finally:
            self._disconnect()

        print(f"\nDone — graph in Neo4j ({self._neo4j_uri}), artifacts in {self.output_dir}/")

    def execute_batched(
        self,
        batch_size: int = 100,
        *,
        skip_evaluate: bool = False,
        skip_export: bool = False,
    ) -> None:
        """Stream documents in batches to avoid loading everything into RAM.

        Unlike ``execute()``, which calls ``preprocess()`` once on all
        documents, this method streams documents from disk in fixed-size
        batches.  Each batch is independently preprocessed (clean, filter,
        dedup, chunk) and streamed to Neo4j via ``build_kg_streaming()``.

        .. warning::
           Cross-batch deduplication and entity resolution are NOT
           performed — each batch is processed independently.  This is a
           deliberate tradeoff for memory efficiency with very large
           datasets.

        Args:
            batch_size: Number of documents to process per batch.
            skip_evaluate: If ``True``, skip the final evaluation step.
            skip_export: If ``True``, skip the final JSON export step.

        Example::

            pipe = MyStreamingPipeline(
                input_paths=["data/huge_corpus/"],
                output_dir="output/",
                clear_db=True,
            )
            pipe.execute_batched(batch_size=200)
        """
        from polygraph.preprocess import chunk, clean, dedup, link, load, quality

        print(f"=== {self.__class__.__name__} (batched, size={batch_size}) ===")
        print(f"Input:  {self.input_paths}")
        print(f"Output: {self.output_dir}")
        print(f"Neo4j:  {self._neo4j_uri}\n")

        self._connect()
        try:
            if self._clear_db:
                print("[neo4j] Clearing database...")
                self._require_builder().clear_database()

            builder = self._require_builder()
            batch_num = 0
            batch: list[Document] = []

            for doc in load.stream(self.input_paths):
                batch.append(doc)
                if len(batch) >= batch_size:
                    batch_num += 1
                    self._process_batch(
                        batch, batch_num, builder, clean, link, quality, dedup, chunk
                    )
                    batch = []

            # Final partial batch
            if batch:
                batch_num += 1
                self._process_batch(batch, batch_num, builder, clean, link, quality, dedup, chunk)

            total_stats = builder.stats
            print(
                f"\n[batched] {batch_num} batch(es) — "
                f"{total_stats['nodes_written']} nodes, "
                f"{total_stats['edges_written']} edges total"
            )

            if not skip_evaluate:
                self.evaluate()
            if not skip_export:
                self.export()

        finally:
            self._disconnect()

        print(f"\nDone — graph in Neo4j ({self._neo4j_uri}), artifacts in {self.output_dir}/")

    def _process_batch(
        self,
        batch: list[Document],
        batch_num: int,
        builder: Neo4jGraphBuilder,
        clean,
        link,
        quality,
        dedup,
        chunk,
    ) -> None:
        """Preprocess and stream a single batch of documents to Neo4j."""
        batch = clean.normalize(batch)
        batch = link.normalize_links(batch)
        batch = quality.filter(batch, min_chars=50, min_words=10)
        batch = dedup.remove_duplicates(batch, method="minhash", threshold=0.85)
        if not batch:
            return

        chunks = chunk.by_sentence(batch, target_tokens=450, overlap_tokens=60)
        chunks = quality.filter(chunks)
        chunks = dedup.remove_duplicates(chunks, method="minhash", threshold=0.85)
        if not chunks:
            return

        self.build_kg_streaming(chunks, builder)
        print(
            f"[batch {batch_num}] {len(batch)} docs → {len(chunks)} chunks → "
            f"Neo4j ({builder.stats['nodes_written']} nodes, "
            f"{builder.stats['edges_written']} edges so far)"
        )
