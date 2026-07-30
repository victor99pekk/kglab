"""Shared-references document-to-document relation extraction.

Extracts reference identifiers (DOIs, arXiv IDs) from each document's text, then
links documents that cite the same external works via ``shares_reference_with``.
"""

import logging
import re
from typing import Any

from polygraph._shared import Document
from polygraph.kg_build.extract.document_relation._base import (
    DocTriple,
    DocumentRelationExtractor,
)
from polygraph.kg_build.extract.document_relation._helpers import (
    doc_entity_id,
    make_document_entities,
)

logger = logging.getLogger(__name__)

_DOI_PATTERN = re.compile(r"\b(10\.\d{4,}/[^\s\]\)},;]+)", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"arxiv:(\d{4}\.\d{4,})", re.IGNORECASE)


class SharedReferencesExtractor(DocumentRelationExtractor):
    """Extract ``shares_reference_with`` relations by comparing the sets of
    referenced works (DOIs + arXiv IDs) between documents.

    Two documents are linked when their Jaccard similarity over extracted
    reference identifiers meets or exceeds *threshold* (default 0.0 — any
    overlap produces an edge).
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        doc_entities = make_document_entities(documents)

        # Extract reference sets per document
        ref_sets: list[tuple[str, set[str]]] = []
        for doc in documents:
            refs: set[str] = set()
            refs.update(_DOI_PATTERN.findall(doc.content))
            refs.update(f"arxiv:{m}" for m in _ARXIV_PATTERN.findall(doc.content))
            if refs:
                ref_sets.append((doc_entity_id(doc), refs))

        # Pairwise Jaccard
        triples: list[DocTriple] = []
        for i in range(len(ref_sets)):
            for j in range(i + 1, len(ref_sets)):
                a_id, a_refs = ref_sets[i]
                b_id, b_refs = ref_sets[j]
                intersection = a_refs & b_refs
                if not intersection:
                    continue
                union = a_refs | b_refs
                jaccard = len(intersection) / len(union)
                if jaccard >= self.threshold:
                    evidence = ", ".join(sorted(intersection)[:3])
                    if len(intersection) > 3:
                        evidence += f" (+{len(intersection) - 3} more)"
                    triples.append((a_id, "shares_reference_with", b_id, evidence, ""))

        logger.info(
            "[SharedReferencesExtractor] %d docs with refs, %d shared-ref edges",
            len(ref_sets),
            len(triples),
        )
        return doc_entities, triples
