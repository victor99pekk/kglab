"""Citation-based document-to-document relation extraction.

Detects citation patterns in document text and matches cited works to other
documents in the batch via DOI, title, or author+year.
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

# DOI: 10.XXXX/...
_DOI_PATTERN = re.compile(r"\b(10\.\d{4,}/[^\s\]\)},;]+)", re.IGNORECASE)

# arXiv: 1234.56789 or arXiv:1234.56789
_ARXIV_PATTERN = re.compile(r"arxiv:(\d{4}\.\d{4,})", re.IGNORECASE)

# Author-year: (Author, 2020), (Author et al., 2020), Author (2020)
_AUTHOR_YEAR_PATTERN = re.compile(
    r"\(([A-Z][a-z]+(?:\s(?:et\s+al\.?|&\s+[A-Z][a-z]+))?),?\s*(\d{4}[a-z]?)\)"
    r"|([A-Z][a-z]+)\s+\((\d{4}[a-z]?)\)",
)


class CitationExtractor(DocumentRelationExtractor):
    """Extract ``cites`` relations by matching citation patterns to other
    documents' metadata.

    Detection order (best match first):
      1. DOI — exact match against another document's ``metadata["doi"]``.
      2. Title — if a citation's surrounding sentence contains another
         document's title (case-insensitive substring).
      3. Author + year — matches ``metadata["authors"]`` and
         ``metadata["year"]`` against author-year citation patterns.
    """

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        doc_entities = make_document_entities(documents)
        doc_index = _build_doc_identity_index(documents)
        triples: list[DocTriple] = []

        for doc in documents:
            subject_id = doc_entity_id(doc)
            content_lower = doc.content.casefold()

            # 1. DOI matches
            for doi in _DOI_PATTERN.findall(doc.content):
                target_id = doc_index.doi.get(doi.casefold())
                if target_id and target_id != subject_id:
                    triples.append((subject_id, "cites", target_id, f"doi:{doi}", doc.doc_id))

            # 2. Title substring matches in citation context
            for doi_or_title, target_id in doc_index.title.items():
                if target_id == subject_id:
                    continue
                if doi_or_title in content_lower:
                    triples.append((subject_id, "cites", target_id, doi_or_title, doc.doc_id))

            # 3. Author-year matches
            for match in _AUTHOR_YEAR_PATTERN.finditer(doc.content):
                author = (match.group(1) or match.group(3) or "").casefold().strip()
                year = (match.group(2) or match.group(4) or "").strip()
                if not author or not year:
                    continue
                target_id = doc_index.author_year.get((author, year))
                if target_id and target_id != subject_id:
                    triples.append(
                        (
                            subject_id,
                            "cites",
                            target_id,
                            f"{match.group(0)}",
                            doc.doc_id,
                        )
                    )

        # Deduplicate by (subject, predicate, object)
        seen: set[tuple[str, str, str]] = set()
        unique: list[DocTriple] = []
        for t in triples:
            key = (t[0], t[1], t[2])
            if key not in seen:
                seen.add(key)
                unique.append(t)

        logger.info(
            "[CitationExtractor] %d docs, %d citation edges",
            len(documents),
            len(unique),
        )
        return doc_entities, unique


# ── Document identity index for citation matching ─────────────


class _DocIdentityIndex:
    """Pre-built lookup structures for matching citations to documents."""

    def __init__(self) -> None:
        self.doi: dict[str, str] = {}  # doi (casefolded) → entity_id
        self.title: dict[str, str] = {}  # title substring (casefolded) → entity_id
        self.author_year: dict[tuple[str, str], str] = {}  # (author, year) → entity_id


def _build_doc_identity_index(documents: list[Document]) -> _DocIdentityIndex:
    """Build lookup index from document metadata for citation matching."""
    idx = _DocIdentityIndex()
    for doc in documents:
        eid = doc_entity_id(doc)

        # DOI
        doi = str(doc.metadata.get("doi", "")).strip()
        if doi:
            idx.doi[doi.casefold()] = eid

        # Title (as substring matcher)
        title = str(doc.metadata.get("title", "")).strip()
        if title and len(title) > 5:
            idx.title[title.casefold()] = eid

        # Author + year
        year = str(doc.metadata.get("year", doc.metadata.get("publish_date", ""))).strip()
        if year and len(year) >= 4:
            year = year[:4]
        authors = doc.metadata.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        first_author = authors[0].strip().casefold() if authors else ""
        if first_author and year:
            idx.author_year[(first_author, year)] = eid

    return idx
