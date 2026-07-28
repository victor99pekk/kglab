"""Semantic chunking — sentence-aligned chunks at topic shift boundaries.

Exports: SemanticChunker
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

from polygraph._shared import Document, Language
from polygraph.preprocess.chunk.fixed import count_tokens
from polygraph.preprocess.chunk.sentence import SentenceChunker


class SemanticChunker(SentenceChunker):
    """Create sentence-aligned chunks at multilingual semantic topic shifts."""

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 60,
        language: Language = Language.ENGLISH,
        similarity_threshold: float = 0.55,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between zero and one")
        super().__init__(target_tokens, overlap_tokens, language)
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.encoder = encoder
        self._embedder: Any | None = None

    def _chunk_one(self, doc: Document) -> list[Document]:
        units = self._expanded_units(doc.content)
        if not units:
            return []
        if len(units) == 1:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(doc.content)
            return [doc]
        embeddings = self._encode(units)
        packed = self._pack_semantic_units(units, embeddings)
        if len(packed) == 1:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(doc.content)
            return [doc]
        return [self._make_chunk(doc, text, index) for index, text in enumerate(packed)]

    def _encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        if self.encoder is not None:
            return self.encoder(texts)
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Semantic chunking requires sentence-transformers. "
                    "Install it with: uv sync --extra embeddings"
                ) from error
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _pack_semantic_units(
        self,
        units: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        if len(embeddings) != len(units):
            raise ValueError("Semantic encoder returned the wrong number of embeddings")
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        minimum_topic_tokens = max(1, self.target_tokens // 3)

        for index, unit in enumerate(units):
            unit_tokens = count_tokens(unit)
            topic_shift = (
                index > 0
                and current_tokens >= minimum_topic_tokens
                and self._cosine(embeddings[index - 1], embeddings[index])
                < self.similarity_threshold
            )
            over_budget = bool(current) and current_tokens + unit_tokens > self.target_tokens
            if current and (topic_shift or over_budget):
                chunks.append(" ".join(current))
                current = self._overlap_units(current)
                current_tokens = sum(count_tokens(item) for item in current)
                if current and current_tokens + unit_tokens > self.target_tokens:
                    current = []
                    current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
        dot = sum(float(left) * float(right) for left, right in zip(first, second, strict=False))
        first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
        second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
        return dot / max(first_norm * second_norm, 1e-12)
