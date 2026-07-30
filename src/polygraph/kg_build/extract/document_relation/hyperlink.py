"""Hyperlink-based document-to-document relation extraction."""

import logging
import re
from typing import Any
from urllib.parse import urlparse

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

_URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")


class HyperlinkExtractor(DocumentRelationExtractor):
    """Extract ``hyperlinks_to`` relations by matching URLs in document content
    to other documents' canonical URLs.
    """

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        doc_entities = make_document_entities(documents)
        url_index = _build_url_index(documents)
        triples: list[DocTriple] = []

        for doc in documents:
            subject_id = doc_entity_id(doc)
            found_urls = _URL_PATTERN.findall(doc.content)
            for url in found_urls:
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
