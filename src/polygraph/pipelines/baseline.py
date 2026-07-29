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
    """Standard pipeline: minhash dedup, spaCy extraction, string resolution.

    Good defaults for getting started. For LLM extraction or embedding
    resolution, subclass and override build_kg().
    """

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
        entities, triples = extract.with_spacy(chunks, ontology=ontology, model="en_core_web_sm")
        resolved = resolve.by_string(entities, threshold=0.85)
        graph = build.from_resolved(resolved, triples)

        print(
            f"[build_kg] {len(entities)} entities → {len(resolved)} resolved, "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return {"graph": graph, "entities": resolved, "triples": triples}
