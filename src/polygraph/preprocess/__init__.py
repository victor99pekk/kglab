"""Pre-processing: raw text → clean, deduplicated chunks.

Function API — each group is a namespace of convenience functions:
    from polygraph.preprocess import load, clean, chunk, quality, dedup

    docs = load.from_paths(["data/"])
    docs = clean.normalize(docs)
    docs = quality.filter(docs, min_chars=50, min_words=10)
    docs = dedup.remove_duplicates(docs, method="minhash")
    chunks = chunk.by_sentence(docs, target_tokens=450, overlap_tokens=60)

Class API — for advanced control:
    from polygraph.preprocess import DataLoader, TextCleaner, ...
"""

from pathlib import Path
from types import SimpleNamespace

from polygraph.preprocess.chunk import SemanticChunker, SentenceChunker, TextChunker
from polygraph.preprocess.clean import TextCleaner
from polygraph.preprocess.dedup import Deduplicator
from polygraph.preprocess.load import DataLoader
from polygraph.preprocess.quality import QualityFilter

# ── Function API ────────────────────────────────────────────────

load = SimpleNamespace()
load.from_paths = lambda paths, formats=None: (
    DataLoader(formats).load([Path(p) for p in paths])
    if formats
    else DataLoader().load([Path(p) for p in paths])
)

clean = SimpleNamespace()
clean.normalize = lambda docs: (
    [TextCleaner().clean(d) for d in docs if TextCleaner().clean(d).content.strip()]
)

quality = SimpleNamespace()
quality.filter = lambda docs, min_chars=50, min_words=10: (
    QualityFilter(min_chars=min_chars, min_words=min_words).filter(docs)
)

chunk = SimpleNamespace()
chunk.by_sentence = lambda docs, target_tokens=450, overlap_tokens=60: (
    SentenceChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens).chunk(docs)
)
chunk.by_fixed = lambda docs, size=500, overlap=100: (
    TextChunker(chunk_size=size, chunk_overlap=overlap).chunk(docs)
)
chunk.by_semantic = (
    lambda docs,
    target_tokens=450,
    overlap_tokens=60,
    threshold=0.55,
    model="paraphrase-multilingual-MiniLM-L12-v2": (
        SemanticChunker(
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            similarity_threshold=threshold,
            model_name=model,
        ).chunk(docs)
    )
)

dedup = SimpleNamespace()
dedup.remove_duplicates = lambda docs, method="minhash", threshold=0.85: (
    Deduplicator(method=method, threshold=threshold).deduplicate(docs)
)

__all__ = [
    "DataLoader",
    "Deduplicator",
    "QualityFilter",
    "SemanticChunker",
    "SentenceChunker",
    "TextChunker",
    "TextCleaner",
    "load",
    "clean",
    "chunk",
    "quality",
    "dedup",
]
