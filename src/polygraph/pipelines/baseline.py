"""Baseline pipeline — standard, dependency-light defaults.

from polygraph.pipelines.baseline import Baseline

pipe = Baseline(input_paths=["data/"], output_dir="output/")
pipe.execute()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polygraph._shared import Document, Ontology
from polygraph._shared.stage_config import (
    BuildConfig,
    DocumentRelationConfig,
    ExtractionConfig,
    ResolutionConfig,
)
from polygraph.kg_build import build, extract, resolve
from polygraph.pipelines._base import Pipeline
from polygraph.preprocess import chunk, clean, dedup, load, quality

_DEFAULT_ONTOLOGY_PATH = Path(__file__).parents[3] / "configs" / "default_ontology.yaml"


class Baseline(Pipeline):
    """Standard pipeline with configurable extraction, resolution, and build methods.

    Accepts optional typed config objects in addition to the raw ``**kwargs``
    dict from the base class. When provided, these are used directly instead
    of dict-digging — the YAML must conform to the code, not the other way around.

    Args:
        input_paths: Data files or directories to load.
        output_dir: Where results are written.
        extraction: Typed extraction config (optional, preferred).
        resolution: Typed resolution config (optional, preferred).
        build: Typed build config (optional, preferred).
        **kwargs: Legacy raw config dict (backward compatible).
    """

    def __init__(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        extraction: ExtractionConfig | None = None,
        resolution: ResolutionConfig | None = None,
        build: BuildConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(input_paths, output_dir, **kwargs)
        self.extraction = extraction
        self.resolution = resolution
        self.build = build

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

    def preprocess(self) -> list[Document]:
        docs = load.from_paths(self.input_paths)
        docs = clean.normalize(docs)
        docs = quality.filter(docs, min_chars=50, min_words=10)
        docs = dedup.remove_duplicates(docs, method="minhash", threshold=0.85)

        # Store raw documents for document-to-document relation extraction
        self._raw_docs = docs

        chunks = chunk.by_sentence(docs, target_tokens=450, overlap_tokens=60)
        chunks = quality.filter(chunks)
        chunks = dedup.remove_duplicates(chunks, method="minhash", threshold=0.85)

        print(f"[preprocess] {len(docs)} documents → {len(chunks)} chunks")
        return chunks

    def build_kg(self, chunks: list[Document]) -> dict[str, Any]:
        ontology = self._load_ontology()

        # ── Resolve extraction config (typed → dict fallback) ──
        ext_cfg = self.extraction or ExtractionConfig.from_dict(self._config.get("extraction"))

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
        if ext_cfg.document_relation.enabled and hasattr(self, "_raw_docs"):
            entities, triples = self._extract_document_relations(
                entities, triples, ext_cfg.document_relation
            )

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

        # ── Entity → Chunk edges (which chunk each entity came from) ──
        entity_chunk_edges: set[tuple[str, str]] = set()
        for triple in triples:
            source_chunk = triple[4] if len(triple) > 4 else ""
            if source_chunk and source_chunk in existing_ids:
                entity_chunk_edges.add((triple[0], source_chunk))
                entity_chunk_edges.add((triple[2], source_chunk))

        for entity_id, chunk_id in entity_chunk_edges:
            triples.append((entity_id, "appears_in", chunk_id, "", chunk_id))

        print(f"[entities] {len(entity_chunk_edges)} entity→chunk edges")

        # ── Resolution ─────────────────────────────────────────
        res_cfg = self.resolution or ResolutionConfig.from_dict(self._config.get("resolution"))
        resolved = resolve.with_method(
            entities,
            method=res_cfg.method,
            threshold=res_cfg.threshold,
            **res_cfg.options,
        )

        # ── Build ──────────────────────────────────────────────
        bld_cfg = self.build or BuildConfig.from_dict(self._config.get("build"))
        graph = build.from_resolved(
            resolved,
            triples,
            method=bld_cfg.method,
            ontology=ontology,
        )

        print(
            f"[build_kg] {len(entities)} entities → {len(resolved)} resolved, "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return {"graph": graph, "entities": resolved, "triples": triples}

    def _extract_document_relations(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple],
        config: DocumentRelationConfig,
    ) -> tuple[list[dict[str, Any]], list[tuple]]:
        """Run document-to-document relation extraction and merge results."""
        from polygraph.kg_build.extract.document_relation import create_doc_relation_method

        methods = config.methods
        extractor = create_doc_relation_method(
            "composite" if len(methods) > 1 else methods[0],
            methods=methods,
            method_options=config.method_options,
        )

        doc_entities, doc_triples = extractor.extract(self._raw_docs)

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
