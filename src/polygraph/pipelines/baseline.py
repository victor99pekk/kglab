"""Baseline pipeline — standard, dependency-light defaults.

from polygraph.pipelines.baseline import Baseline

pipe = Baseline(input_paths=["data/"], output_dir="output/")
pipe.run()
"""

from polygraph._shared import Document
from polygraph.kg_build import build, extract, resolve
from polygraph.pipelines._base import Pipeline
from polygraph.preprocess import chunk, clean, dedup, load, quality


class Baseline(Pipeline):
    """Standard pipeline: minhash dedup, spaCy extraction, string resolution.

    Good defaults for getting started. For LLM extraction or embedding
    resolution, subclass and override build_kg().
    """

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
        entities, triples = extract.with_spacy(chunks, model="en_core_web_sm")
        resolved = resolve.by_string(entities, threshold=0.85)
        graph = build.from_resolved(resolved, triples)

        print(
            f"[build_kg] {len(entities)} entities → {len(resolved)} resolved, "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return {"graph": graph, "entities": resolved, "triples": triples}
