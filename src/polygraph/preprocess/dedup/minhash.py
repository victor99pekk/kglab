"""MinHash LSH near-duplicate detection.

Exports: GlobalDeduplicator, DuplicateAssignment, DuplicateMatch
"""

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = None  # type: ignore[assignment,misc]
    MinHashLSH = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

MINHASH_PERMUTATIONS = 128
LSH_THRESHOLD_DEFAULT = 0.85


@dataclass(frozen=True)
class DuplicateAssignment:
    """Duplicate decision retained for curation audit records."""

    cluster_id: str | None
    canonical_id: str | None
    is_duplicate: bool
    method: str | None = None
    similarity: float | None = None
    matched_record_id: str | None = None


@dataclass(frozen=True)
class DuplicateMatch:
    """Direct evidence that connected two records in a duplicate cluster."""

    record_id: str
    matched_record_id: str
    method: str
    similarity: float


class GlobalDeduplicator:
    """Cluster exact and surface-level near duplicates with MinHash candidates.

    ``shingle_fn`` keeps the historic character-trigram behaviour by default,
    while allowing curation to supply language-aware word shingles without
    changing the KG pipeline.
    """

    def __init__(
        self,
        threshold: float = LSH_THRESHOLD_DEFAULT,
        num_perm: int = MINHASH_PERMUTATIONS,
        shingle_fn: Callable[[str], set[str]] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Deduplication threshold must be between 0 and 1.")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_fn = shingle_fn or self._grams
        self.last_matches: list[DuplicateMatch] = []

    def cluster(self, records: Sequence[dict[str, object]]) -> dict[str, DuplicateAssignment]:
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

        matches: dict[tuple[int, int], DuplicateMatch] = {}
        exact: dict[str, int] = {}
        signatures: list[object] = []
        gram_sets: list[set[str]] = []
        for index, record in enumerate(records):
            content = str(record["content"])
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            grams = self.shingle_fn(content)
            gram_sets.append(grams)
            if content_hash in exact:
                existing = exact[content_hash]
                union(index, existing)
                self._record_match(matches, records, index, existing, "exact_hash", 1.0)
            else:
                exact[content_hash] = index
            signatures.append(self._minhash(grams))

        if MinHash is not None and MinHashLSH is not None:
            lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
            for index, signature in enumerate(signatures):
                lsh.insert(str(index), signature)
            for index, signature in enumerate(signatures):
                for candidate in lsh.query(signature):
                    candidate_index = int(candidate)
                    if candidate_index < index:
                        similarity = self._jaccard(gram_sets[index], gram_sets[candidate_index])
                        if similarity >= self.threshold:
                            union(index, candidate_index)
                            self._record_match(
                                matches,
                                records,
                                index,
                                candidate_index,
                                "minhash_jaccard",
                                similarity,
                            )
        else:
            for index, grams in enumerate(gram_sets):
                for candidate_index, candidate_grams in enumerate(gram_sets[:index]):
                    similarity = len(grams & candidate_grams) / max(len(grams | candidate_grams), 1)
                    if similarity >= self.threshold:
                        union(index, candidate_index)
                        self._record_match(
                            matches,
                            records,
                            index,
                            candidate_index,
                            "ngram_jaccard_fallback",
                            similarity,
                        )

        groups: dict[int, list[int]] = {}
        for index in range(len(records)):
            groups.setdefault(find(index), []).append(index)
        self.last_matches = sorted(
            matches.values(),
            key=lambda match: (match.record_id, match.matched_record_id, match.method),
        )
        return self._assignments(records, groups, self.last_matches)

    def _minhash(self, shingles: set[str]) -> object:
        if MinHash is None:
            return shingles
        signature = MinHash(num_perm=self.num_perm)
        for gram in shingles:
            signature.update(gram.encode("utf-8"))
        return signature

    @staticmethod
    def _grams(text: str) -> set[str]:
        return {text[index : index + 3] for index in range(max(len(text) - 2, 1))}

    @staticmethod
    def _jaccard(first: set[str], second: set[str]) -> float:
        return len(first & second) / max(len(first | second), 1)

    @staticmethod
    def _record_match(
        matches: dict[tuple[int, int], DuplicateMatch],
        records: Sequence[dict[str, object]],
        first: int,
        second: int,
        method: str,
        similarity: float,
    ) -> None:
        key = tuple(sorted((first, second)))
        current = matches.get(key)
        if current and current.similarity >= similarity:
            return
        matches[key] = DuplicateMatch(
            record_id=str(records[first]["doc_id"]),
            matched_record_id=str(records[second]["doc_id"]),
            method=method,
            similarity=similarity,
        )

    @staticmethod
    def _assignments(
        records: Sequence[dict[str, object]],
        groups: dict[int, list[int]],
        matches: Sequence[DuplicateMatch],
    ) -> dict[str, DuplicateAssignment]:
        assignments: dict[str, DuplicateAssignment] = {}
        matches_by_record: dict[str, list[DuplicateMatch]] = {}
        for match in matches:
            matches_by_record.setdefault(match.record_id, []).append(match)
            matches_by_record.setdefault(match.matched_record_id, []).append(match)
        for members in groups.values():
            canonical_index = sorted(
                members,
                key=lambda i: (-float(records[i]["quality_score"]), str(records[i]["doc_id"])),
            )[0]
            canonical_id = str(records[canonical_index]["doc_id"])
            cluster_id = (
                f"dup-{min(str(records[i]['doc_id']) for i in members)}"
                if len(members) > 1
                else None
            )
            for index in members:
                record_id = str(records[index]["doc_id"])
                evidence = sorted(
                    matches_by_record.get(record_id, []),
                    key=lambda match: (-match.similarity, match.method, match.matched_record_id),
                )[0:1]
                assignments[record_id] = DuplicateAssignment(
                    cluster_id=cluster_id,
                    canonical_id=canonical_id,
                    is_duplicate=len(members) > 1 and index != canonical_index,
                    method=evidence[0].method if evidence else None,
                    similarity=evidence[0].similarity if evidence else None,
                    matched_record_id=(
                        evidence[0].matched_record_id
                        if evidence and evidence[0].record_id == record_id
                        else evidence[0].record_id
                        if evidence
                        else None
                    ),
                )
        return assignments
