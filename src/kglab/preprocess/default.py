"""DefaultPreprocessor — config-driven preprocessing using the function registries.

Reads ``PreprocessConfig`` and dispatches each stage via the existing
``kglab.preprocess`` auto-discovered registries (``load``, ``clean``,
``chunk``, ``quality``, ``dedup``, ``link``).

Tier 1 (simple knobs): runs the fixed default sequence, dispatching on
    ``chunk_method``, ``dedup_method``, etc.

Tier 2 (stage list): runs user-defined stages in order, skipping disabled
    ones and forwarding ``options`` as keyword arguments.

.. note::
    Stage registries are accessed lazily (via ``_get_registries()``) to
    avoid a circular import with ``kglab.preprocess.__init__``, which
    imports this module before the ``_set`` convenience aliases are
    registered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kglab._shared import (
    Document,
    PreprocessResult,
    discover_pipeline_languages,
    filter_by_language,
)
from kglab._shared.stage_config import PreprocessConfig, PreprocessStage
from kglab.preprocess._base import Preprocessor

# ── Pipeline-level language support ────────────────────────────
# Auto-computed from folder structure (mirrors logic previously in
# pipelines/baseline.py).  Stages with an ``en/`` subdirectory are
# English-only; stages without language subdirectories are universal.

logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        ontology: Any | None = None,
        extraction: Any | None = None,
    ) -> None:
        self.config = config or PreprocessConfig()
        self.ontology = ontology
        self.extraction = extraction  # ExtractionConfig
        self._raw: list[Document] = []
        self._entities: list[dict[str, Any]] = []
        self._triples: list[tuple] = []
        self._extra: dict[str, Any] = {}

    @property
    def _r(self) -> Any:
        """Lazily-loaded stage registries (avoids circular import)."""
        from kglab.preprocess import chunk, clean, dedup, link, load, quality

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

    def preprocess(self, input_paths: list[Path]) -> PreprocessResult:
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
            return PreprocessResult()

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
        return PreprocessResult(
            chunks=chunks,
            raw_docs=self._raw,
            entities=self._entities,
            triples=self._triples,
            extra=self._extra,
        )

    # ── Tier 2: User-defined stage list ────────────────────────

    def _run_custom_stages(self, input_paths: list[Path]) -> PreprocessResult:
        """Run user-defined stages in order, skipping disabled ones."""
        state: list[Document] = []
        for stage in self.config.stages:  # type: ignore[union-attr]
            if not stage.enabled:
                continue
            state = self._dispatch_stage(stage, state, input_paths)
        return PreprocessResult(
            chunks=state,
            raw_docs=self._raw,
            entities=self._entities,
            triples=self._triples,
            extra=self._extra,
        )

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

        elif name == "extract":
            return self._run_extract_stage(docs, method, opts)

        elif name == "extra":
            # Store arbitrary key-value pairs for downstream use
            self._extra[method] = opts
            return docs

        elif name == "link":
            self._link_accumulated_entities(method, opts)
            return docs

        else:
            # Unknown stages pass through — custom Preprocessor subclasses
            # can handle them by overriding _dispatch_stage
            print(f"[preprocess] Unknown stage '{name}' — passing through")
            return docs
            known = ", ".join(
                [
                    "load",
                    "language_filter",
                    "clean",
                    "link_normalize",
                    "quality",
                    "dedup",
                    "chunk",
                    "extract",
                    "extra",
                    "link",
                ]
            )
            raise ValueError(f"Unknown preprocessing stage: '{name}'. Known stages: {known}")

    # ── Extraction stage ──────────────────────────────────────

    def _run_extract_stage(
        self, docs: list[Document], method: str, opts: dict[str, Any]
    ) -> list[Document]:
        """Run entity/relation extraction on the current document set.

        Accumulates extracted entities and triples into ``self._entities``
        and ``self._triples`` so ``build_kg`` can skip re-extraction.
        Returns the documents unchanged (extraction doesn't alter docs).
        """
        if not docs:
            return docs
        if self.ontology is None:
            raise ValueError(
                "Extraction stage requires an ontology. "
                "Pass ontology= to DefaultPreprocessor or ensure the pipeline "
                "provides one."
            )

        from kglab._shared.stage_config import ExtractionConfig
        from kglab.kg_build import extract as kg_extract

        ext_cfg = self.extraction or ExtractionConfig()

        if ext_cfg.mode == "joint":
            entities, triples = kg_extract.jointly(
                docs,
                self.ontology,
                method=method or ext_cfg.joint_method,
                options={**ext_cfg.options, **opts},
            )
        else:
            entities, triples = kg_extract.with_methods(
                docs,
                self.ontology,
                entity_method=method or ext_cfg.entity_method,
                relation_method=opts.get("relation_method", ext_cfg.relation_method),
                entity_options=ext_cfg.entity_options,
                relation_options=ext_cfg.relation_options,
            )

        self._entities.extend(entities)
        self._triples.extend(triples)
        print(
            f"[extract] {len(entities)} entities, {len(triples)} triples "
            f"(total: {len(self._entities)} entities, {len(self._triples)} triples)"
        )
        return docs

    # ── Entity linking stage ───────────────────────────────────

    def _link_accumulated_entities(self, method: str, opts: dict[str, Any]) -> None:
        """Link accumulated entities to an external KB in-place.

        Only runs if entities have been extracted during preprocessing
        (via ``"extract"`` stages).  Enriches ``self._entities`` with
        ``kb_id`` and ``kb_source`` fields.
        """
        if not self._entities:
            logger.debug("[link] No accumulated entities to link — skipping")
            return

        from kglab.kg_build import link as kg_link

        self._entities = kg_link.entities(
            self._entities,
            method=method,
            **opts,
        )

    # ── Memory management ──────────────────────────────────────

    def free_raw_docs(self) -> None:
        """Release raw documents to free memory after ``build_kg``.

        Call this after document-to-document relation extraction
        is complete to avoid holding both raw docs and chunks in RAM.
        """
        self._raw = []
