"""Baseline pipeline — standard, dependency-light defaults.

from polygraph.pipelines.baseline import Baseline

# Zero config (sensible defaults):
pipe = Baseline(input_paths=["data/"], output_dir="output/")
pipe.execute()

# With preprocessing knobs:
pipe = Baseline(
    input_paths=["data/"],
    output_dir="output/",
    preprocess=PreprocessConfig(chunk_method="semantic", chunk_target_tokens=300),
)
pipe.execute()

# Full stage control:
pipe = Baseline(
    input_paths=["data/"],
    output_dir="output/",
    preprocess=PreprocessConfig(stages=[
        PreprocessStage("load", "baseline"),
        PreprocessStage("chunk", "sentence", options={"target_tokens": 500}),
    ]),
)
pipe.execute()

# Custom preprocessor:
pipe = Baseline(
    input_paths=["data/"],
    output_dir="output/",
    preprocessor=MyCustomPreprocessor(),
)
pipe.execute()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from polygraph._shared import Document, Ontology
from polygraph._shared.stage_config import (
    BuildConfig,
    DocumentRelationConfig,
    EvalConfig,
    ExportConfig,
    ExtractionConfig,
    LinkingConfig,
    PreprocessConfig,
    ResolutionConfig,
)
from polygraph.kg_build import build_kg_into, extract, resolve
from polygraph.kg_build.build.sqlite import SQLiteGraphWriter
from polygraph.kg_build.build.writer import NetworkXGraphWriter
from polygraph.kg_export.graph_store import (
    GraphStore,
    Neo4jGraphStore,
    NetworkXGraphStore,
    SQLiteGraphStore,
    create_graph_store,
)
from polygraph.kg_export.neo4j.builder import Neo4jGraphBuilder
from polygraph.pipelines._base import Pipeline
from polygraph.preprocess import DefaultPreprocessor
from polygraph.preprocess._base import Preprocessor

_DEFAULT_ONTOLOGY_PATH = Path(__file__).parents[3] / "configs" / "default_ontology.yaml"


def _entity_chunk_membership_triples(
    entities: list[dict[str, Any]],
    valid_chunk_ids: set[str],
) -> list[tuple[str, str, str, str, str]]:
    """Build direct Entity→Chunk membership edges from extraction provenance."""
    memberships: set[tuple[str, str]] = set()
    for entity in entities:
        entity_id = entity.get("id", "")
        if not entity_id or entity.get("type") in {"Chunk", "Document"}:
            continue

        source_chunk_ids = entity.get("source_chunk_ids", [])
        if isinstance(source_chunk_ids, str):
            source_chunk_ids = [source_chunk_ids]
        for chunk_id in source_chunk_ids:
            if chunk_id in valid_chunk_ids:
                memberships.add((entity_id, chunk_id))

    return [
        (entity_id, "appears_in", chunk_id, "", chunk_id)
        for entity_id, chunk_id in sorted(memberships)
    ]


class Baseline(Pipeline):
    """Standard pipeline with configurable extraction, resolution, and build methods.

    Preprocessing is delegated to a ``Preprocessor`` instance — by default
    ``DefaultPreprocessor``, which is driven by ``PreprocessConfig``.
    Pass a custom ``Preprocessor`` subclass to replace the entire
    preprocessing strategy.

    Args:
        input_paths: Data files or directories to load.
        output_dir: Where results are written.
        preprocessor: Custom ``Preprocessor`` instance (Tier 3 — full control).
            Overrides ``preprocess`` config when both are provided.
        preprocess: ``PreprocessConfig`` for simple knobs (Tier 1) or
            stage-level control (Tier 2). Ignored when ``preprocessor``
            is provided.
        extraction: Typed extraction config (optional).
        resolution: Typed resolution config (optional).
        build: Typed build config (optional).
        eval_: Evaluation config (optional).
        export: Export config (optional).
        **kwargs: Legacy raw config dict (backward compatible).
    """

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        preprocess: PreprocessConfig | None = None,
        extraction: ExtractionConfig | None = None,
        resolution: ResolutionConfig | None = None,
        build: BuildConfig | None = None,
        linking: LinkingConfig | None = None,
        eval_: EvalConfig | None = None,
        export: ExportConfig | None = None,
        graph_store_backend: str | None = None,
        graph_store_options: dict[str, Any] | None = None,
        graph_store: GraphStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Wire up the preprocessor, passing ontology and extraction config
        # so the "extract" stage can use them during preprocessing.
        if preprocessor is None:
            ontology = self._load_ontology()
            self._preprocessor = DefaultPreprocessor(
                preprocess, ontology=ontology, extraction=extraction
            )
        else:
            self._preprocessor = preprocessor
        self.extraction = extraction
        self.resolution = resolution
        self.build = build
        self.linking = linking
        self.eval_config = eval_
        self.export_config = export
        self.graph_store_backend = graph_store_backend
        self.graph_store_options = graph_store_options or {}
        self._target_store = graph_store

    # ── Language support (delegated) ───────────────────────────

    @property
    def supported_languages(self) -> set[str]:
        """Languages fully supported by every stage in this pipeline."""
        return self._preprocessor.supported_languages

    # ── Internal helpers ───────────────────────────────────────

    def _load_ontology(self) -> Ontology:
        """Load the ontology from the path given at construction time."""
        ontology_path = self._config.get("ontology_path")
        if ontology_path:
            return Ontology.from_yaml(Path(ontology_path))
        if _DEFAULT_ONTOLOGY_PATH.exists():
            return Ontology.from_yaml(_DEFAULT_ONTOLOGY_PATH)
        raise FileNotFoundError(
            "No ontology found. Pass ontology_path= to the pipeline or place "
            f"a YAML file at {_DEFAULT_ONTOLOGY_PATH}"
        )

    # ── Stage methods ──────────────────────────────────────────

    def preprocess(self) -> list[Document]:
        """Delegate to the configured preprocessor."""
        return self._preprocessor.preprocess(self.input_paths)

    def build_kg(
        self,
        chunks: list[Document] | Any,
        *,
        existing_store: GraphStore | None = None,
        graph_store: GraphStore | None = None,
    ) -> dict[str, Any]:
        """Build the knowledge graph from chunks or a PreprocessResult.

        If ``chunks`` is a ``PreprocessResult`` and contains pre-extracted
        entities/triples, extraction is skipped and those are used directly.

        Args:
            existing_store: Optional ``GraphStore`` of an existing KG to
                continue building on top of (new content is merged in).
            graph_store: Optional pre-configured ``GraphStore`` to build into
                and return — e.g. ``Neo4jGraphStore(uri=..., user=..., ...)``
                so connection credentials are set once instead of per-call
                ``graph_store_options``.

        Returns:
            Dict with ``graph``, ``entities``, ``triples``, and
            ``graph_store`` (in-memory ``NetworkXGraphStore`` by default).
        """
        from polygraph._shared.types import PreprocessResult

        ontology = self._load_ontology()
        existing_store = (
            existing_store if existing_store is not None else getattr(self, "_existing_store", None)
        )
        target_store = (
            graph_store if graph_store is not None else getattr(self, "_target_store", None)
        )
        ext_cfg = self.extraction or ExtractionConfig.from_dict(self._config.get("extraction"))

        # ── Unpack PreprocessResult if provided ────────────────
        pre_extracted_entities: list[dict[str, Any]] = []
        pre_extracted_triples: list[tuple] = []
        extra: dict[str, Any] = {}
        if isinstance(chunks, PreprocessResult):
            pre_extracted_entities = chunks.entities
            pre_extracted_triples = chunks.triples
            extra = chunks.extra
            chunks = chunks.chunks

        # ── Extract entities and triples (or use pre-extracted) ─
        if pre_extracted_entities:
            print(
                f"[build_kg] Using {len(pre_extracted_entities)} pre-extracted entities "
                f"and {len(pre_extracted_triples)} triples from preprocessing"
            )
            entities = list(pre_extracted_entities)
            triples = list(pre_extracted_triples)
        else:
            if ext_cfg.mode == "joint":
                entities, triples = extract.jointly(
                    chunks,
                    ontology,
                    method=ext_cfg.joint_method,
                    options=ext_cfg.options,
                )
            elif ext_cfg.mode == "composed":
                entity_options = ext_cfg.entity_options
                if entity_options is None and ext_cfg.entity_method == "spacy":
                    entity_options = {"model_name": "en_core_web_sm"}
                entities, triples = extract.with_methods(
                    chunks,
                    ontology,
                    entity_method=ext_cfg.entity_method,
                    relation_method=ext_cfg.relation_method,
                    entity_options=entity_options,
                    relation_options=ext_cfg.relation_options,
                )
            else:
                raise ValueError("extraction.mode must be one of: composed, joint")

        # ── Document-to-document relation extraction ───────────
        raw_docs = self._preprocessor.raw_docs
        if ext_cfg.document_relation.enabled and raw_docs:
            entities, triples = self._extract_document_relations(
                entities, triples, ext_cfg.document_relation, raw_docs=raw_docs
            )
            # Free raw docs early to avoid holding both raw docs and chunks in RAM
            if hasattr(self._preprocessor, "free_raw_docs"):
                self._preprocessor.free_raw_docs()

        # ── Add chunk nodes and connect to parent documents ────
        from polygraph.kg_build.extract.document_relation._helpers import (
            doc_entity_id,
            make_chunk_entities,
        )

        chunk_entities = make_chunk_entities(chunks)
        existing_ids = {e["id"] for e in entities}
        for ce in chunk_entities:
            if ce["id"] not in existing_ids:
                entities.append(ce)
                existing_ids.add(ce["id"])

        # Link each chunk to its parent Document node
        for ch in chunks:
            parent_doc_id = ch.metadata.get("parent_doc_id", ch.source)
            parent_node_id = doc_entity_id(
                Document(
                    content="",
                    source=ch.source,
                    doc_id=parent_doc_id,
                )
            )
            triples.append((ch.doc_id, "part_of", parent_node_id, "", ch.doc_id))

        # Link consecutive chunks within the same document
        chunk_next_edges = 0
        for i in range(len(chunks) - 1):
            c1 = chunks[i]
            c2 = chunks[i + 1]
            parent1 = c1.metadata.get("parent_doc_id") or c1.source
            parent2 = c2.metadata.get("parent_doc_id") or c2.source
            if parent1 and parent1 == parent2:
                triples.append((c1.doc_id, "next", c2.doc_id, "", c1.doc_id))
                chunk_next_edges += 1

        print(
            f"[chunks] {len(chunk_entities)} chunk nodes, "
            f"{len(chunks)} chunk→document edges, "
            f"{chunk_next_edges} chunk→next_chunk edges"
        )

        # ── Resolution ─────────────────────────────────────────
        res_cfg = self.resolution or ResolutionConfig.from_dict(self._config.get("resolution"))
        resolved, entity_id_map = resolve.with_method_and_mapping(
            entities,
            method=res_cfg.method,
            threshold=res_cfg.threshold,
            **res_cfg.options,
        )
        triples = [
            (
                entity_id_map.get(triple[0], triple[0]),
                triple[1],
                entity_id_map.get(triple[2], triple[2]),
                *triple[3:],
            )
            for triple in triples
        ]

        # ── Entity linking (enrich with external KB IDs) ───────
        link_cfg = self.linking or LinkingConfig.from_dict(self._config.get("linking"))
        if link_cfg.enabled:
            from polygraph.kg_build import link as kg_link

            resolved = kg_link.entities(
                resolved,
                method=link_cfg.method,
                **link_cfg.options,
            )

        # ── Entity → Chunk edges (which chunks each entity came from) ──
        entity_chunk_triples = _entity_chunk_membership_triples(
            resolved,
            {chunk.doc_id for chunk in chunks},
        )
        triples.extend(entity_chunk_triples)

        print(f"[entities] {len(entity_chunk_triples)} entity→chunk edges")

        # ── Build ──────────────────────────────────────────────
        bld_cfg = self.build or BuildConfig.from_dict(self._config.get("build"))
        backend = self.graph_store_backend
        if backend in (None, "auto"):
            if isinstance(target_store, Neo4jGraphStore):
                backend = "neo4j"
            else:
                backend = "sqlite" if bld_cfg.method == "sqlite" else "networkx"

        if backend == "neo4j":
            # Stream the KG directly into Neo4j — no in-memory graph.
            graph, store, neo4j_stats = self._build_kg_streaming_neo4j(
                chunks, resolved, triples, target_store=target_store
            )
            print(
                f"[build_kg] streamed {len(resolved)} entities / {len(triples)} triples "
                f"to Neo4j ({neo4j_stats.get('nodes_written', 0)} nodes, "
                f"{neo4j_stats.get('edges_written', 0)} edges)"
            )
        else:
            # One shared build path for every backend: write through a
            # GraphWriter via build_kg_into.  Chunk/Document nodes and
            # PART_OF/NEXT edges are derived from `chunks`; only entity
            # nodes and semantic/APPEARS_IN triples are passed in.  Swap the
            # writer to change where the graph is stored — the routine is
            # storage-agnostic.
            if bld_cfg.method == "sqlite":
                writer = SQLiteGraphWriter(
                    db_path=str(self.output_dir / "knowledge_graph.db"),
                    ontology=ontology,
                )
            else:
                writer = NetworkXGraphWriter(ontology=ontology)
            build_kg_into(
                writer,
                chunks,
                [e for e in resolved if e.get("type") not in ("Chunk", "Document")],
                [t for t in triples if t[1] not in ("part_of", "next")],
            )
            graph = writer.graph

            # Continue building on top of an existing KG when one is provided.
            if existing_store is not None:
                if bld_cfg.method == "sqlite":
                    raise NotImplementedError(
                        "existing_store is currently supported for the default "
                        "in-memory backend only (build method 'networkx')."
                    )
                graph = self._merge_existing_graph(graph, existing_store)

            # Expose the built graph through the storage-agnostic GraphStore.
            store = self._build_graph_store(graph, bld_cfg.method, resolved, triples)
            neo4j_stats = None

        print(
            f"[build_kg] {len(entities)} entities → {len(resolved)} resolved, "
            f"{graph.number_of_nodes() if graph is not None else 0} nodes, "
            f"{graph.number_of_edges() if graph is not None else 0} edges"
        )
        result: dict[str, Any] = {"graph": graph, "entities": resolved, "triples": triples}
        if extra:
            result["extra"] = extra
        if neo4j_stats is not None:
            result["neo4j_stats"] = neo4j_stats

        self._graph_store = store
        result["graph_store"] = store
        return result

    # ── GraphStore helpers ─────────────────────────────────────

    def _build_graph_store(
        self,
        graph: Any,
        method: str,
        entities: list[dict[str, Any]],
        triples: list[tuple],
    ) -> GraphStore:
        """Create the ``GraphStore`` for the built graph.

        The storage backend is controlled by ``graph_store_backend``:

        * ``None`` (default) — match the build method: in-memory
          ``NetworkXGraphStore``, or ``SQLiteGraphStore`` for sqlite builds.
        * ``"networkx"`` — in-memory representation (the built graph).
        * ``"sqlite"`` — file-backed store (requires ``build.method="sqlite"``).
        * ``"json"`` / ``"graphml"`` — export to a file and read it back.

        ``"neo4j"`` is handled separately by ``_build_kg_streaming_neo4j``,
        which streams the graph directly into Neo4j during the build stage.
        """
        backend = self.graph_store_backend
        dict(self.graph_store_options)

        if backend in (None, "auto"):
            backend = "sqlite" if method == "sqlite" else "networkx"

        if backend == "networkx":
            return NetworkXGraphStore(graph)
        if backend == "sqlite":
            if method != "sqlite":
                raise ValueError(
                    "graph_store_backend='sqlite' requires build.method='sqlite' "
                    "(an in-memory graph cannot back a SQLiteGraphStore)."
                )
            return SQLiteGraphStore(graph)
        if backend == "json":
            path = self._export_graph_file(graph, entities, triples, "json")
            return create_graph_store("json", path=path)
        if backend == "graphml":
            path = self._export_graph_file(graph, entities, triples, "graphml")
            return create_graph_store("graphml", path=path)
        raise ValueError(
            f"Unknown graph_store_backend '{backend}'. "
            "Available: networkx, sqlite, neo4j, json, graphml"
        )

    def _build_kg_streaming_neo4j(
        self,
        chunks: list[Document],
        resolved: list[dict[str, Any]],
        triples: list[tuple],
        target_store: GraphStore | None = None,
    ) -> tuple[None, GraphStore, dict[str, int]]:
        """Stream the KG directly into Neo4j — no in-memory graph.

        Writes through the same shared ``build_kg_into`` routine used by the
        in-memory path, but via ``Neo4jGraphBuilder``.  Chunk/Document nodes
        and ``PART_OF`` / ``NEXT`` edges are derived from ``chunks``; only
        entity nodes and semantic / ``APPEARS_IN`` triples are passed in.

        Connection credentials come from ``target_store`` (a pre-configured
        ``Neo4jGraphStore``) when provided, otherwise from
        ``graph_store_options``.

        Returns:
            ``(graph=None, graph_store, stats)`` — the graph lives in Neo4j.
        """
        from polygraph.kg_export.neo4j.upload import _get_connection

        options = dict(self.graph_store_options)
        clear = bool(options.pop("clear", False))

        # Reuse the credentials of a pre-configured store (set once), so the
        # pipeline caller never repeats uri/user/password as arguments.
        if isinstance(target_store, Neo4jGraphStore):
            connection: dict[str, Any] = {
                "uri": target_store.uri,
                "user": target_store.user,
                "password": target_store.password,
            }
        else:
            connection = options

        entity_nodes = [e for e in resolved if e.get("type") not in ("Chunk", "Document")]
        semantic_triples = [t for t in triples if t[1] not in ("part_of", "next")]

        driver = _get_connection(**connection)
        stats: dict[str, int] = {}
        try:
            with driver.session() as session:
                builder = Neo4jGraphBuilder(session, ontology=self._load_ontology())
                if clear:
                    builder.clear_database()
                build_kg_into(builder, chunks, entity_nodes, semantic_triples)
                stats = builder.stats
        finally:
            driver.close()

        store: GraphStore = (
            target_store
            if isinstance(target_store, Neo4jGraphStore)
            else create_graph_store("neo4j", **connection)
        )
        return None, store, stats

    def _export_graph_file(
        self,
        graph: Any,
        entities: list[dict[str, Any]],
        triples: list[tuple],
        fmt: str,
    ) -> Path:
        """Export the built graph to ``output_dir`` (JSON or GraphML)."""
        from polygraph.kg_export.json.exporter import GraphExporter

        self.output_dir.mkdir(parents=True, exist_ok=True)
        GraphExporter().export(graph, entities, triples, output_dir=self.output_dir, formats=[fmt])
        suffix = "json" if fmt == "json" else "graphml"
        return self.output_dir / f"knowledge_graph.{suffix}"

    def _merge_existing_graph(self, graph: Any, existing_store: GraphStore) -> Any:
        """Merge a newly built graph onto an existing KG from a ``GraphStore``.

        The existing graph is materialised (``to_networkx()``) and the new
        nodes and edges are merged in — combining edge predicates, evidence
        sentences, source chunks, and weights instead of duplicating edges.
        """
        existing = existing_store.to_networkx()
        if not isinstance(existing, nx.DiGraph):
            raise TypeError(
                "existing_store must materialise to an nx.DiGraph "
                "(e.g. NetworkXGraphStore or JSONGraphStore) to continue building."
            )

        for node, data in graph.nodes(data=True):
            if node in existing:
                existing.nodes[node].update(data)
            else:
                existing.add_node(node, **data)

        for u, v, data in graph.edges(data=True):
            if existing.has_edge(u, v):
                edge = existing.edges[u, v]
                for pred in data.get("predicates", []):
                    if pred not in edge.get("predicates", []):
                        edge["predicates"] = edge.get("predicates", []) + [pred]
                edge["weight"] = len(edge.get("predicates", []))
                for key in ("source_texts", "source_chunk_ids"):
                    for item in data.get(key, []):
                        if item not in edge.get(key, []):
                            edge[key] = edge.get(key, []) + [item]
                for rel in data.get("relations", []):
                    if rel not in edge.get("relations", []):
                        edge["relations"] = edge.get("relations", []) + [rel]
                if data.get("description") and not edge.get("description"):
                    edge["description"] = data["description"]
            else:
                existing.add_edge(u, v, **data)

        return existing

    def execute(
        self,
        input_paths: list[str | Path] | None = None,
        output_dir: str | Path | None = None,
        cache: bool = False,
        force: bool = False,
        existing_store: GraphStore | None = None,
        graph_store_backend: str | None = None,
        graph_store_options: dict[str, Any] | None = None,
        graph_store: GraphStore | None = None,
    ) -> dict[str, Any]:
        """Full pipeline, forwarding export config and paths.

        Args:
            existing_store: An existing ``GraphStore`` to continue building
                on top of — new nodes/edges are merged into it instead of
                starting from an empty graph.
            graph_store_backend: Storage backend for the returned
                ``"graph_store"``: ``None`` (auto, default), ``"networkx"``,
                ``"sqlite"``, ``"neo4j"``, ``"json"``, or ``"graphml"``.
                Overrides the constructor value for this run.
            graph_store_options: Extra kwargs for the store backend (e.g.
                ``uri`` / ``user`` / ``password`` / ``clear`` for Neo4j).
            graph_store: A pre-configured ``GraphStore`` to build into and
                return (e.g. ``Neo4jGraphStore(uri=..., user=..., ...)``) so
                connection credentials are set once instead of per-call.

        Returns:
            The built KG dict (see ``Pipeline.execute``) — including
            ``"graph_store"`` (in-memory ``NetworkXGraphStore`` by default).
        """
        if existing_store is not None:
            self._existing_store = existing_store
        if graph_store is not None:
            self._target_store = graph_store
        if graph_store_backend is not None:
            self.graph_store_backend = graph_store_backend
        if graph_store_options is not None:
            self.graph_store_options = graph_store_options
        return super().execute(
            input_paths=input_paths,
            output_dir=output_dir,
            export_config=self.export_config,
            cache=cache,
            force=force,
        )

    def _build_run_manifest(self, results_summary: dict[str, Any]) -> Any:
        """Populate the run manifest with all Baseline stage configs.

        Resolves defaults for any config not explicitly passed at construction
        time so the manifest always captures the effective configuration.
        """
        from dataclasses import asdict

        from polygraph._shared.stage_config import (
            BuildConfig,
            EvalConfig,
            ExportConfig,
            ExtractionConfig,
            LinkingConfig,
            ResolutionConfig,
        )

        manifest = super()._build_run_manifest(results_summary)

        # ── Preprocessing config ──
        if hasattr(self._preprocessor, "config"):
            manifest.preprocess = asdict(self._preprocessor.config)

        # ── Extraction config (resolve defaults if not provided) ──
        ext_cfg = self.extraction or ExtractionConfig.from_dict(self._config.get("extraction"))
        manifest.extraction = asdict(ext_cfg)

        # ── Resolution config ──
        res_cfg = self.resolution or ResolutionConfig.from_dict(self._config.get("resolution"))
        manifest.resolution = asdict(res_cfg)

        # ── Build config ──
        bld_cfg = self.build or BuildConfig.from_dict(self._config.get("build"))
        manifest.build = asdict(bld_cfg)

        # ── Linking config ──
        link_cfg = self.linking or LinkingConfig.from_dict(self._config.get("linking"))
        manifest.linking = asdict(link_cfg)

        # ── Evaluation config ──
        eval_cfg = self.eval_config or EvalConfig.from_dict(self._config.get("evaluation"))
        manifest.evaluation = asdict(eval_cfg)

        # ── Export config ──
        export_cfg = self.export_config or ExportConfig.from_dict(self._config.get("export"))
        manifest.export = asdict(export_cfg)

        # ── Ontology path ──
        ontology_path = self._config.get("ontology_path")
        if ontology_path:
            manifest.ontology = {"path": str(ontology_path)}

        return manifest

    def _extract_document_relations(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple],
        config: DocumentRelationConfig,
        raw_docs: list[Document] | None = None,
    ) -> tuple[list[dict[str, Any]], list[tuple]]:
        """Run document-to-document relation extraction and merge results."""
        from polygraph.kg_build.extract.document_relation import create_doc_relation_method

        docs = raw_docs or []
        if not docs:
            return entities, triples

        methods = config.methods
        if len(methods) > 1:
            extractor = create_doc_relation_method(
                "composite", methods=methods, method_options=config.method_options
            )
        else:
            extractor = create_doc_relation_method(methods[0])

        doc_entities, doc_triples = extractor.extract(docs)

        existing_ids = {e["id"] for e in entities}
        for de in doc_entities:
            if de["id"] not in existing_ids:
                entities.append(de)
                existing_ids.add(de["id"])

        triples.extend(doc_triples)

        print(
            f"[document_relation] {len(doc_entities)} doc nodes, "
            f"{len(doc_triples)} doc-doc edges "
            f"({', '.join(methods)})"
        )
        return entities, triples
