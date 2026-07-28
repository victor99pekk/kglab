"""KG generation: extract entities → resolve duplicates → build graph.

Each stage is a folder under kg_build/.  Every .py file in a stage folder
registers one method.  Drop a new file to add a method — auto-discovered.

Function API:
    from polygraph.kg_build import extract, resolve, build

    entities, triples = extract.with_spacy(chunks, model="en_core_web_sm")
    resolved = resolve.by_string(entities, threshold=0.85)
    graph = build.from_resolved(resolved, triples)

Class API — for advanced control:
    from polygraph.kg_build import EnglishExtractor, EntityResolver, GraphBuilder, ...
"""

from types import SimpleNamespace

from polygraph.kg_build.build import GraphBuilder
from polygraph.kg_build.build.default import build_graph
from polygraph.kg_build.extract.entities import (
    EnglishExtractor,
    Entity,
    EntityExtractor,
    SimpleExtractor,
)
from polygraph.kg_build.extract.graphgen import GraphGenExtractor
from polygraph.kg_build.extract.relations import RelationExtractor
from polygraph.kg_build.resolve import EntityResolver
from polygraph.kg_build.resolve.embedding import resolve_embedding
from polygraph.kg_build.resolve.string import resolve_string

# ── Function API ────────────────────────────────────────────────

extract = SimpleNamespace()


def _extract_spacy(chunks, model="en_core_web_sm"):
    eng = EnglishExtractor(model_name=model)
    rel = RelationExtractor()
    all_entities = []
    all_triples = []
    for c in chunks:
        ents = eng.extract(c.content)
        trips = rel.extract(c.content, ents, source_chunk_id=c.doc_id)
        all_entities.extend(e.to_dict() for e in ents)
        all_triples.extend(trips)
    return all_entities, all_triples


def _extract_graphgen(chunks, model="deepseek-v4-pro", max_gleanings=3):
    gen = GraphGenExtractor(model_name=model, max_gleanings=max_gleanings)
    all_entities = []
    all_triples = []
    for c in chunks:
        ents, trips = gen.extract(c.content, source_chunk_id=c.doc_id)
        all_entities.extend(e.to_dict() for e in ents)
        all_triples.extend(trips)
    return all_entities, all_triples


extract.with_spacy = _extract_spacy
extract.with_graphgen = _extract_graphgen

resolve = SimpleNamespace()
resolve.by_string = lambda entities, threshold=0.85: resolve_string(entities, threshold)
resolve.by_embedding = lambda entities, threshold=0.85: resolve_embedding(entities, threshold)

build = SimpleNamespace()
build.from_resolved = lambda resolved, triples: build_graph(resolved, triples)

__all__ = [
    "EnglishExtractor",
    "Entity",
    "EntityExtractor",
    "EntityResolver",
    "GraphBuilder",
    "GraphGenExtractor",
    "RelationExtractor",
    "SimpleExtractor",
    "extract",
    "resolve",
    "build",
]
