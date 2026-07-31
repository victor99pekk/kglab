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

from polygraph._shared import Document, Ontology
from polygraph._shared.stage_config import (
    BuildConfig,
    DocumentRelationConfig,
    EvalConfig,
    ExportConfig,
    ExtractionConfig,
    PreprocessConfig,
    ResolutionConfig,
)
from polygraph.kg_build import build, extract, resolve
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
        input_paths: list[str | Path],
        output_dir: str | Path,
        preprocessor: Preprocessor | None = None,
        preprocess: PreprocessConfig | None = None,
        extraction: ExtractionConfig | None = None,
        resolution: ResolutionConfig | None = None,
        build: BuildConfig | None = None,
        eval_: EvalConfig | None = None,
        export: ExportConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_paths, output_dir, **kwargs)
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
        self.eval_config = eval_
        self.export_config = export

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

    def build_kg(self, chunks: list[Document] | Any) -> dict[str, Any]:
        """Build the knowledge graph from chunks or a PreprocessResult.

        If ``chunks`` is a ``PreprocessResult`` and contains pre-extracted
        entities/triples, extraction is skipped and those are used directly.
        """
        from polygraph._shared.types import PreprocessResult

        ontology = self._load_ontology()
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

        # ── Entity → Chunk edges (which chunks each entity came from) ──
        entity_chunk_triples = _entity_chunk_membership_triples(
            resolved,
            {chunk.doc_id for chunk in chunks},
        )
        triples.extend(entity_chunk_triples)

        print(f"[entities] {len(entity_chunk_triples)} entity→chunk edges")

        # ── Build ──────────────────────────────────────────────
        bld_cfg = self.build or BuildConfig.from_dict(self._config.get("build"))
        build_kwargs: dict[str, Any] = {}
        if bld_cfg.method == "sqlite":
            build_kwargs["db_path"] = str(self.output_dir / "knowledge_graph.db")
        graph = build.from_resolved(
            resolved,
            triples,
            method=bld_cfg.method,
            ontology=ontology,
            **build_kwargs,
        )

        print(
            f"[build_kg] {len(entities)} entities → {len(resolved)} resolved, "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        result: dict[str, Any] = {"graph": graph, "entities": resolved, "triples": triples}
        if extra:
            result["extra"] = extra
        return result

    def execute(self, cache: bool = False, force: bool = False) -> None:
        """Full pipeline, forwarding eval/export configs."""
        super().execute(
            eval_config=self.eval_config,
            export_config=self.export_config,
            cache=cache,
            force=force,
        )

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
