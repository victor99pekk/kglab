"""Embedding-based semantic duplicate clustering.

Exports: SemanticDeduplicator
"""

import math
from collections.abc import Callable, Sequence

from polygraph.preprocess.dedup.minhash import DuplicateAssignment, DuplicateMatch


class SemanticDeduplicator:
    """Embedding-based semantic duplicate clustering for small and medium datasets.

    This is an opt-in semantic baseline inspired by SemDeDup. It evaluates all
    document pairs, so it is intentionally capped and is not a web-scale ANN
    implementation.
    """

    def __init__(
        self,
        threshold: float = 0.92,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        max_records: int = 5_000,
        encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Semantic deduplication threshold must be between 0 and 1.")
        self.threshold = threshold
        self.model_name = model_name
        self.max_records = max_records
        self.encoder = encoder
        self.last_matches: list[DuplicateMatch] = []

    def cluster(self, records: Sequence[dict[str, object]]) -> dict[str, DuplicateAssignment]:
        if len(records) > self.max_records:
            raise ValueError(
                f"Semantic deduplication supports at most {self.max_records} records per run; "
                "use MinHash or add an ANN index for larger datasets."
            )
        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)

        embeddings = self._encode([str(record["content"]) for record in records])
        matches: list[DuplicateMatch] = []
        for index, embedding in enumerate(embeddings):
            for candidate_index, candidate_embedding in enumerate(embeddings[:index]):
                similarity = self._cosine_similarity(embedding, candidate_embedding)
                if similarity >= self.threshold:
                    union(index, candidate_index)
                    matches.append(
                        DuplicateMatch(
                            record_id=str(records[index]["doc_id"]),
                            matched_record_id=str(records[candidate_index]["doc_id"]),
                            method="semantic_cosine",
                            similarity=similarity,
                        )
                    )
        groups: dict[int, list[int]] = {}
        for index in range(len(records)):
            groups.setdefault(find(index), []).append(index)

        # Build assignments (same logic as GlobalDeduplicator)
        from polygraph.preprocess.dedup.minhash import GlobalDeduplicator

        return GlobalDeduplicator._assignments(records, groups, matches)

    def _encode(self, texts: list[str]) -> Sequence[Sequence[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Semantic deduplication requires sentence-transformers. "
                "Install it with: uv sync --extra embeddings"
            ) from error
        model = SentenceTransformer(self.model_name)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
        dot_product = sum(left * right for left, right in zip(first, second, strict=False))
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        return dot_product / max(first_norm * second_norm, 1e-12)
