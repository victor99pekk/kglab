"""DefaultPreprocessor — config-driven preprocessing using the function registries.

Reads ``PreprocessConfig`` and dispatches each stage via the existing
``polygraph.preprocess`` auto-discovered registries (``load``, ``clean``,
``chunk``, ``quality``, ``dedup``, ``link``).

Tier 1 (simple knobs): runs the fixed default sequence, dispatching on
    ``chunk_method``, ``dedup_method``, etc.

Tier 2 (stage list): runs user-defined stages in order, skipping disabled
    ones and forwarding ``options`` as keyword arguments.

.. note::
    Stage registries are accessed lazily (via ``_get_registries()``) to
    avoid a circular import with ``polygraph.preprocess.__init__``, which
    imports this module before the ``_set`` convenience aliases are
    registered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polygraph._shared import (
    Document,
    discover_pipeline_languages,
    filter_by_language,
)
from polygraph._shared.stage_config import PreprocessConfig, PreprocessStage
from polygraph.preprocess._base import Preprocessor

# ── Pipeline-level language support ────────────────────────────
# Auto-computed from folder structure (mirrors logic previously in
# pipelines/baseline.py).  Stages with an ``en/`` subdirectory are
# English-only; stages without language subdirectories are universal.

_PREPROCESS_ROOT = Path(__file__).parents[1] / "preprocess"

_PIPELINE_LANGUAGES = discover_pipeline_languages(
    _PREPROCESS_ROOT / "clean",
    _PREPROCESS_ROOT / "chunk",
)


class DefaultPreprocessor(Preprocessor):
    """Config-driven preprocessing using the existing function registries.

    This is the default preprocessing strategy used by ``Baseline``.
    When constructed with no config, it runs the same sequence as the
    current ``Baseline.preprocess()`` with all defaults.

    Args:
        config: Preprocessing configuration.  ``None`` uses all defaults
            (Tier 0).
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._raw: list[Document] = []

    @property
    def _r(self) -> Any:
        """Lazily-loaded stage registries (avoids circular import)."""
        from polygraph.preprocess import chunk, clean, dedup, link, load, quality

        return type(
            "_R",
            (),
            {
                "load": load,
                "clean": clean,
                "chunk": chunk,
                "quality": quality,
                "dedup": dedup,
                "link": link,
            },
        )()

    # ── Preprocessor contract ──────────────────────────────────

    @property
    def supported_languages(self) -> set[str]:
        """Languages fully supported by every constituent stage."""
        return _PIPELINE_LANGUAGES

    @property
    def raw_docs(self) -> list[Document]:
        """Raw documents retained for downstream doc-relation extraction.

        Returns an empty list if preprocessing hasn't run yet.  Set to
        an empty list *after* ``build_kg`` to free memory early.
        """
        return self._raw

    def preprocess(self, input_paths: list[Path]) -> list[Document]:
        """Run preprocessing — Tier 2 stages or Tier 1 defaults."""
        if self.config.stages is not None:
            return self._run_custom_stages(input_paths)
        return self._run_default_stages(input_paths)

    # ── Tier 1: Default sequence with simple-knob dispatch ─────

    def _run_default_stages(self, input_paths: list[Path]) -> list[Document]:
        """Run the fixed default sequence, dispatching on simple knobs."""
        c = self.config
        r = self._r
        docs = r.load.from_paths(input_paths)

        # Language gate
        kept, skipped = filter_by_language(docs, self.supported_languages)
        if skipped:
            skipped_ids = [d.doc_id for d in skipped]
            langs = {d.language for d in skipped}
            print(
                f"[language] Skipping {len(skipped)} document(s) — "
                f"languages {sorted(langs)} not supported. "
                f"Pipeline supports: {sorted(self.supported_languages)}"
            )
            print(f"[language] Skipped IDs: {skipped_ids}")
        docs = kept

        if not docs:
            print("[preprocess] No supported documents — pipeline stopping.")
            return []

        if c.clean_enabled:
            docs = r.clean.normalize(docs)
        if c.link_normalize_enabled:
            docs = r.link.normalize_links(docs)
        docs = r.quality.filter(docs, min_chars=c.quality_min_chars, min_words=c.quality_min_words)
        docs = r.dedup.remove_duplicates(
            docs, method=c.doc_dedup_method, threshold=c.doc_dedup_threshold
        )

        # Retain raw docs for downstream document-relation extraction.
        # Caller can set self._raw = [] after build_kg to free memory.
        self._raw = docs

        # Chunk dispatch
        if c.chunk_method == "sentence":
            chunks = r.chunk.by_sentence(
                docs,
                target_tokens=c.chunk_target_tokens,
                overlap_tokens=c.chunk_overlap_tokens,
            )
        elif c.chunk_method == "fixed":
            chunks = r.chunk.by_fixed(
                docs,
                size=c.chunk_target_tokens,
                overlap=c.chunk_overlap_tokens,
            )
        elif c.chunk_method == "semantic":
            chunks = r.chunk.by_semantic(
                docs,
                target_tokens=c.chunk_target_tokens,
                overlap_tokens=c.chunk_overlap_tokens,
                threshold=c.chunk_semantic_threshold,
                model=c.chunk_semantic_model,
            )
        else:
            available = "sentence, fixed, semantic"
            raise ValueError(f"Unknown chunk_method '{c.chunk_method}'. Available: {available}")

        chunks = r.quality.filter(chunks)
        chunks = r.dedup.remove_duplicates(
            chunks, method=c.chunk_dedup_method, threshold=c.chunk_dedup_threshold
        )

        print(f"[preprocess] {len(docs)} documents → {len(chunks)} chunks")
        return chunks

    # ── Tier 2: User-defined stage list ────────────────────────

    def _run_custom_stages(self, input_paths: list[Path]) -> list[Document]:
        """Run user-defined stages in order, skipping disabled ones."""
        state: list[Document] = []
        for stage in self.config.stages:  # type: ignore[union-attr]
            if not stage.enabled:
                continue
            state = self._dispatch_stage(stage, state, input_paths)
        return state

    def _dispatch_stage(
        self,
        stage: PreprocessStage,
        docs: list[Document],
        input_paths: list[Path],
    ) -> list[Document]:
        """Map stage name → preprocess registry call."""
        r = self._r
        name = stage.name
        method = stage.method
        opts = stage.options or {}

        if name == "load":
            result = r.load.from_paths(input_paths, **opts)
            kept, skipped = filter_by_language(result, self.supported_languages)
            if skipped:
                skipped_ids = [d.doc_id for d in skipped]
                langs = {d.language for d in skipped}
                print(
                    f"[language] Skipping {len(skipped)} document(s) — "
                    f"languages {sorted(langs)} not supported."
                )
                print(f"[language] Skipped IDs: {skipped_ids}")
            return kept

        elif name == "language_filter":
            kept, skipped = filter_by_language(docs, self.supported_languages)
            if skipped:
                print(f"[language] Skipped {len(skipped)} document(s).")
            return kept

        elif name == "clean":
            if not docs:
                return docs
            return r.clean.normalize(docs)

        elif name == "link_normalize":
            if not docs:
                return docs
            if method == "none":
                return docs
            return r.link.normalize_links(docs)

        elif name == "quality":
            if not docs:
                return docs
            return r.quality.filter(docs, **opts)

        elif name == "dedup":
            if not docs:
                return docs
            return r.dedup.remove_duplicates(
                docs,
                method=method,
                threshold=opts.get("threshold", 0.85),
            )

        elif name == "chunk":
            if not docs:
                return docs
            # Store raw docs before chunking
            self._raw = docs
            chunk_fn = getattr(r.chunk, f"by_{method}")
            return chunk_fn(docs, **opts)

        else:
            known = ", ".join(
                [
                    "load",
                    "language_filter",
                    "clean",
                    "link_normalize",
                    "quality",
                    "dedup",
                    "chunk",
                ]
            )
            raise ValueError(f"Unknown preprocessing stage: '{name}'. Known stages: {known}")

    # ── Memory management ──────────────────────────────────────

    def free_raw_docs(self) -> None:
        """Release raw documents to free memory after ``build_kg``.

        Call this after document-to-document relation extraction
        is complete to avoid holding both raw docs and chunks in RAM.
        """
        self._raw = []
