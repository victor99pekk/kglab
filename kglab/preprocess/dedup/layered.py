"""Layered deduplication — combines multiple methods (minhash → semantic).

Exports: Deduplicator
"""

import hashlib
import logging
from collections.abc import Callable, Sequence

from kglab._shared import Document
from kglab.preprocess.dedup.minhash import (
    LSH_THRESHOLD_DEFAULT,
    MINHASH_PERMUTATIONS,
    GlobalDeduplicator,
)
from kglab.preprocess.dedup.semantic import SemanticDeduplicator

logger = logging.getLogger(__name__)


class Deduplicator:
    """Detects and removes near-duplicate documents.

    Supports multiple methods selectable by name:
    none, exact, minhash, simhash, ngram, semantic, layered
    """

    def __init__(
        self,
        threshold: float = LSH_THRESHOLD_DEFAULT,
        method: str = "minhash",
        num_perm: int = MINHASH_PERMUTATIONS,
        semantic_threshold: float | None = None,
        semantic_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        semantic_max_records: int = 5_000,
        semantic_encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.method = method
        self.num_perm = num_perm
        self.semantic_threshold = (
            semantic_threshold if semantic_threshold is not None else threshold
        )
        self.semantic_model = semantic_model
        self.semantic_max_records = semantic_max_records
        self.semantic_encoder = semantic_encoder

    def deduplicate(self, documents: list[Document]) -> list[Document]:
        """Return deduplicated list of documents."""
        if not documents:
            return []

        if self.method == "none":
            return list(documents)
        if self.method == "exact":
            return self._exact_dedup(documents)
        if self.method == "minhash":
            return self._global_minhash_dedup(documents)
        elif self.method == "simhash":
            return self._simhash_dedup(documents)
        elif self.method == "ngram":
            return self._ngram_dedup(documents)
        elif self.method == "semantic":
            return self._semantic_dedup(documents)
        elif self.method == "layered":
            return self._semantic_dedup(self._global_minhash_dedup(documents))
        raise ValueError(
            f"Unknown dedup method '{self.method}'. Choose one of: "
            "none, exact, minhash, simhash, ngram, semantic, layered"
        )

    def _global_minhash_dedup(self, documents: list[Document]) -> list[Document]:
        records = [
            {"doc_id": str(index), "content": document.content, "quality_score": 0.0}
            for index, document in enumerate(documents)
        ]
        assignments = GlobalDeduplicator(self.threshold, self.num_perm).cluster(records)
        kept = [
            document
            for index, document in enumerate(documents)
            if not assignments[str(index)].is_duplicate
        ]
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"MinHash dedup: removed {removed} near-duplicate documents")
        return kept

    def _simhash_dedup(self, documents: list[Document]) -> list[Document]:
        seen: list[int] = []
        kept: list[Document] = []
        for doc in documents:
            sig = self._simhash(doc.content)
            is_dup = False
            for existing in seen:
                if self._hamming_distance(sig, existing) <= self._simhash_distance_threshold():
                    is_dup = True
                    break
            if not is_dup:
                seen.append(sig)
                kept.append(doc)
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"SimHash dedup: removed {removed} near-duplicate documents")
        return kept

    def _ngram_dedup(self, documents: list[Document]) -> list[Document]:
        kept: list[Document] = []
        seen_sets: list[set[str]] = []
        for doc in documents:
            ngrams = self._char_ngrams(doc.content, n=5)
            is_dup = False
            for existing in seen_sets:
                jaccard = len(ngrams & existing) / max(len(ngrams | existing), 1)
                if jaccard >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_sets.append(ngrams)
                kept.append(doc)
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"N-gram dedup: removed {removed} near-duplicate documents")
        return kept

    def _exact_dedup(self, documents: list[Document]) -> list[Document]:
        seen: set[str] = set()
        kept: list[Document] = []
        for doc in documents:
            h = hashlib.sha256(doc.content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                kept.append(doc)
        removed = len(documents) - len(kept)
        if removed:
            logger.info(f"Exact dedup: removed {removed} exact duplicate documents")
        return kept

    def _semantic_dedup(self, documents: list[Document]) -> list[Document]:
        if len(documents) > self.semantic_max_records:
            raise ValueError(
                f"Semantic dedup cannot run on {len(documents)} records: it exceeds "
                f"semantic_max_records ({self.semantic_max_records}). Raise "
                "semantic_max_records or process in smaller batches — the semantic "
                "layer is never silently skipped."
            )
        records = [
            {"doc_id": str(index), "content": document.content, "quality_score": 0.0}
            for index, document in enumerate(documents)
        ]
        engine = SemanticDeduplicator(
            threshold=self.semantic_threshold,
            model_name=self.semantic_model,
            max_records=self.semantic_max_records,
            encoder=self.semantic_encoder,
        )
        assignments = engine.cluster(records)
        kept = [
            document
            for index, document in enumerate(documents)
            if not assignments[str(index)].is_duplicate
        ]
        removed = len(documents) - len(kept)
        if removed:
            logger.info("Semantic dedup: removed %d duplicate documents", removed)
        return kept

    def _tokenize(self, text: str, n: int = 3) -> list[str]:
        return [text[i : i + n] for i in range(max(len(text) - n + 1, 1))]

    def _char_ngrams(self, text: str, n: int = 5) -> set[str]:
        return {text[i : i + n] for i in range(max(len(text) - n + 1, 1))}

    def _simhash(self, text: str) -> int:
        tokens = self._tokenize(text, n=4)
        v = [0] * 64
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for j in range(64):
                if h & (1 << j):
                    v[j] += 1
                else:
                    v[j] -= 1
        return sum((1 << i) for i in range(64) if v[i] > 0)

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _simhash_distance_threshold(self) -> int:
        if self.threshold >= 0.95:
            return 2
        elif self.threshold >= 0.85:
            return 3
        elif self.threshold >= 0.75:
            return 4
        return 6
