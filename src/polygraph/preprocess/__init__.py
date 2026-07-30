"""Pre-processing: raw text → clean, deduplicated chunks.

Each stage is a folder under preprocess/.  Every .py file in a stage folder
registers one method.  Drop a new file to add a method — auto-discovered.

Function API:
    from polygraph.preprocess import load, clean, chunk, quality, dedup

    docs = load.from_paths(["data/"])          # load/baseline.py
    docs = clean.normalize(docs)               # clean/baseline.py
    docs = quality.filter(docs, min_chars=50)  # quality/baseline.py
    docs = dedup.remove_duplicates(docs, method="minhash")  # dedup/*.py
    chunks = chunk.by_sentence(docs)           # chunk/sentence.py
    chunks = chunk.by_fixed(docs)              # chunk/fixed.py
    chunks = chunk.by_semantic(docs)           # chunk/semantic.py

Class API:
    from polygraph.preprocess import DataLoader, TextCleaner, ...
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from polygraph.preprocess.chunk import SemanticChunker, SentenceChunker, TextChunker
from polygraph.preprocess.clean import EnglishCleaner, TextCleaner, TextCleanerBackend
from polygraph.preprocess.dedup import (
    Deduplicator,
    DuplicateAssignment,
    DuplicateMatch,
    GlobalDeduplicator,
    SemanticDeduplicator,
)
from polygraph.preprocess.link import normalize_links
from polygraph.preprocess.load import DataLoader
from polygraph.preprocess.quality import (
    QualityFilter,
    QualityProfile,
    QualityProfiler,
    QualityThresholds,
)

# ═══════════════════════════════════════════════════════════════
# Backend registry
# ═══════════════════════════════════════════════════════════════


class BackendRegistry:
    """Callable registry for stage methods — auto-discovered from folder."""

    def __init__(self, name: str = "stage") -> None:
        self._name = name
        self._backends: dict[str, Any] = {}

    def register(self, name: str):
        """Decorator: register a function as a named method."""

        def decorator(fn):
            self._backends[name] = fn
            return fn

        return decorator

    def _set(self, name: str, fn) -> None:
        self._backends[name] = fn

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._backends[name]
        except KeyError:
            available = ", ".join(sorted(self._backends))
            raise AttributeError(
                f"No '{self._name}' method '{name}'. Available: {available}"
            ) from None

    def list(self) -> list[str]:
        return sorted(self._backends)

    def __repr__(self) -> str:
        return f"<{self._name} methods: {self.list()}>"


def _discover_stage_methods(pkg_name: str, stage: str, registry: BackendRegistry) -> None:
    """Scan a stage folder and register all public functions as methods."""
    stage_path = Path(__file__).parent / stage
    if not stage_path.is_dir():
        return

    for py_file in sorted(stage_path.glob("*.py")):
        if py_file.stem.startswith("_") or py_file.stem == "__init__":
            continue
        mod = importlib.import_module(f"{pkg_name}.{stage}.{py_file.stem}")
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                registry._set(name, obj)


# ── Public registries per stage ─────────────────────────────────

load = BackendRegistry("load")
clean = BackendRegistry("clean")
link = BackendRegistry("link")
chunk = BackendRegistry("chunk")
quality = BackendRegistry("quality")
dedup = BackendRegistry("dedup")


# ── Auto-discover methods from stage folders ────────────────────

_discover_stage_methods("polygraph.preprocess", "load", load)
_discover_stage_methods("polygraph.preprocess", "clean", clean)
_discover_stage_methods("polygraph.preprocess", "link", link)
_discover_stage_methods("polygraph.preprocess", "chunk", chunk)
_discover_stage_methods("polygraph.preprocess", "quality", quality)
_discover_stage_methods("polygraph.preprocess", "dedup", dedup)


# ── Register convenience aliases (backward-compatible) ─────────

load._set(
    "from_paths",
    lambda paths, formats=None: (
        DataLoader(formats).load([Path(p) for p in (paths if isinstance(paths, list) else [paths])])
        if formats
        else DataLoader().load([Path(p) for p in (paths if isinstance(paths, list) else [paths])])
    ),
)

clean._set(
    "normalize",
    lambda docs: ([TextCleaner().clean(d) for d in docs if TextCleaner().clean(d).content.strip()]),
)

quality._set(
    "filter",
    lambda docs, min_chars=50, min_words=10: (
        QualityFilter(min_chars=min_chars, min_words=min_words).filter(docs)
    ),
)

chunk._set(
    "by_sentence",
    lambda docs, target_tokens=450, overlap_tokens=60: (
        SentenceChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens).chunk(docs)
    ),
)
chunk._set(
    "by_fixed",
    lambda docs, size=500, overlap=100: (
        TextChunker(chunk_size=size, chunk_overlap=overlap).chunk(docs)
    ),
)
chunk._set(
    "by_semantic",
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
    ),
)

dedup._set(
    "remove_duplicates",
    lambda docs, method="minhash", threshold=0.85: (
        Deduplicator(method=method, threshold=threshold).deduplicate(docs)
    ),
)


__all__ = [
    "DataLoader",
    "Deduplicator",
    "DuplicateAssignment",
    "DuplicateMatch",
    "EnglishCleaner",
    "GlobalDeduplicator",
    "QualityFilter",
    "QualityProfiler",
    "QualityProfile",
    "QualityThresholds",
    "SemanticChunker",
    "SemanticDeduplicator",
    "SentenceChunker",
    "TextChunker",
    "TextCleaner",
    "TextCleanerBackend",
    "chunk",
    "clean",
    "dedup",
    "link",
    "load",
    "normalize_links",
    "quality",
]
