"""Shared-authors document-to-document relation extraction."""

import logging
from typing import Any

from kglab._shared import Document
from kglab.kg_build.extract.document_relation._base import (
    DocTriple,
    DocumentRelationExtractor,
)
from kglab.kg_build.extract.document_relation._helpers import (
    doc_entity_id,
    make_document_entities,
)

logger = logging.getLogger(__name__)


class SharedAuthorsExtractor(DocumentRelationExtractor):
    """Extract ``shares_authors_with`` relations by comparing author metadata
    across documents.

    Expects documents to carry an ``authors`` field in ``metadata`` — either a
    list of strings or a comma-separated string. Authors are normalized
    (lowercased, stripped) before comparison.
    """

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        doc_entities = make_document_entities(documents)
        author_map: dict[str, set[str]] = {}
        for doc in documents:
            authors = self._parse_authors(doc)
            if authors:
                author_map[doc_entity_id(doc)] = authors

        triples: list[DocTriple] = []
        doc_ids = list(author_map.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                overlap = author_map[doc_ids[i]] & author_map[doc_ids[j]]
                if overlap:
                    triples.append(
                        (
                            doc_ids[i],
                            "shares_authors_with",
                            doc_ids[j],
                            ", ".join(sorted(overlap)),
                            "",
                        )
                    )

        logger.info(
            "[SharedAuthorsExtractor] %d docs with authors, %d shared-author edges",
            len(author_map),
            len(triples),
        )
        return doc_entities, triples

    @staticmethod
    def _parse_authors(doc: Document) -> set[str] | None:
        authors = doc.metadata.get("authors")
        if not authors:
            return None
        if isinstance(authors, list):
            items = [str(a).strip().casefold() for a in authors if a]
        elif isinstance(authors, str):
            items = [a.strip().casefold() for a in authors.split(",") if a.strip()]
        else:
            return None
        cleaned = {a for a in items if a}
        return cleaned if cleaned else None
