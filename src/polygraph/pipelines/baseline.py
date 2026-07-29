"""Baseline pipeline — standard, dependency-light defaults.

from polygraph.pipelines.baseline import Baseline

pipe = Baseline(input_paths=["data/"], output_dir="output/")
pipe.execute()
"""

from pathlib import Path

from polygraph._shared import Document, Ontology
from polygraph.kg_build import build, extract, resolve
from polygraph.pipelines._base import Pipeline
from polygraph.preprocess import chunk, clean, dedup, load, quality

_DEFAULT_ONTOLOGY_PATH = Path(__file__).parents[3] / "configs" / "default_ontology.yaml"


class Baseline(Pipeline):
    """Standard pipeline with configurable extraction, resolution, and build methods."""

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

        chunks = chunk.by_sentence(docs, target_tokens=450, overlap_tokens=60)
        chunks = quality.filter(chunks)
        chunks = dedup.remove_duplicates(chunks, method="minhash", threshold=0.85)

        print(f"[preprocess] {len(docs)} documents → {len(chunks)} chunks")
        return chunks

    def build_kg(self, chunks: list[Document]) -> dict:
        ontology = self._load_ontology()
        extraction = self._config.get("extraction", {}) or {}
        legacy_method = extraction.get("method")
        mode = extraction.get(
            "mode",
            "joint" if legacy_method == "graphgen" else "composed",
        )

        if mode == "joint":
            entities, triples = extract.jointly(
                chunks,
                ontology,
                method=extraction.get("joint_method", "graphgen"),
                options=extraction.get("options", {}),
            )
        elif mode == "composed":
            entity_method = extraction.get("entity_method", "spacy")
            entity_options = extraction.get("entity_options")
            if entity_options is None:
                entity_options = (
                    {"model_name": "en_core_web_sm"} if entity_method == "spacy" else {}
                )
            entities, triples = extract.with_methods(
                chunks,
                ontology,
                entity_method=entity_method,
                relation_method=extraction.get("relation_method", "ontology_rules"),
                entity_options=entity_options,
                relation_options=extraction.get("relation_options", {}),
            )
        else:
            raise ValueError("pipeline.extraction.mode must be one of: composed, joint")

        resolution = self._config.get("resolution", {}) or {}
        resolved = resolve.with_method(
            entities,
            method=resolution.get("method", "string"),
            threshold=resolution.get("threshold", 0.85),
            **resolution.get("options", {}),
        )

        graph_config = self._config.get("build", {}) or {}
        graph = build.from_resolved(
            resolved,
            triples,
            method=graph_config.get("method", "networkx"),
            ontology=ontology,
        )

        print(
            f"[build_kg] {len(entities)} entities → {len(resolved)} resolved, "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return {"graph": graph, "entities": resolved, "triples": triples}
