"""Hyperlink-based document-to-document relation extraction.

Reads canonical ``metadata["outgoing_urls"]`` (populated by
:func:`~kglab.preprocess.link.normalize.normalize_links`) and
matches them against other documents' ``metadata["url"]``.
"""

import logging
from typing import Any
from urllib.parse import urlparse

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


class HyperlinkExtractor(DocumentRelationExtractor):
    """Extract ``hyperlinks_to`` relations by matching ``outgoing_urls`` on
    each document to other documents' canonical URLs.

    Requires that :func:`~kglab.preprocess.link.normalize.normalize_links`
    has been run during preprocessing — this populates the canonical
    ``metadata["outgoing_urls"]`` field from either structured input fields
    or content parsing.
    """

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        doc_entities = make_document_entities(documents)
        url_index = _build_url_index(documents)
        triples: list[DocTriple] = []

        for doc in documents:
            subject_id = doc_entity_id(doc)
            outgoing_urls: list[str] = doc.metadata.get("outgoing_urls", [])
            for url in outgoing_urls:
                normalized = _normalize_url(url)
                target_id = url_index.get(normalized)
                if target_id and target_id != subject_id:
                    triples.append((subject_id, "hyperlinks_to", target_id, url, doc.doc_id))

        logger.info("[HyperlinkExtractor] %d docs, %d hyperlinks", len(documents), len(triples))
        return doc_entities, triples


def _build_url_index(documents: list[Document]) -> dict[str, str]:
    """Map normalized URLs → document entity IDs."""
    index: dict[str, str] = {}
    for doc in documents:
        doc_url = doc.metadata.get("url", "")
        if doc_url:
            index[_normalize_url(doc_url)] = doc_entity_id(doc)
        if doc.doc_id and doc.doc_id.startswith("http"):
            index[_normalize_url(doc.doc_id)] = doc_entity_id(doc)
    return index


def _normalize_url(url: str) -> str:
    """Normalize a URL for matching: strip trailing slash, fragment, lowercase host."""
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized
