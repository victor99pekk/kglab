"""Link normalization — extract and canonicalize document hyperlinks.

Exports: normalize_links
"""

from __future__ import annotations

import logging
import re

from kglab._shared import Document

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")

_DEFAULT_URL_FIELDS = ["links", "references", "citations", "outgoing_urls"]


def normalize_links(
    docs: list[Document],
    url_fields: list[str] | None = None,
) -> list[Document]:
    """Ensure every document has ``metadata["outgoing_urls"]``.

    Discovers URLs from structured metadata fields first (e.g.,
    ``"links"``, ``"references"``, ``"citations"``), then falls back to
    parsing the document content text.  This lets the downstream
    ``HyperlinkExtractor`` read a single canonical field regardless of
    whether links were provided explicitly or embedded in prose.

    Args:
        docs: Documents to enrich.
        url_fields: Metadata keys to check for structured URL lists or
            strings.  Defaults to ``["links", "references", "citations",
            "outgoing_urls"]``.

    Returns:
        The same list of documents, with ``metadata["outgoing_urls"]``
        set on each.
    """
    fields = url_fields or _DEFAULT_URL_FIELDS

    for doc in docs:
        urls: set[str] = set()

        # 1. Try structured metadata fields
        for field in fields:
            value = doc.metadata.get(field)
            if isinstance(value, list):
                urls.update(str(v) for v in value if v)
            elif isinstance(value, str):
                urls.update(_URL_PATTERN.findall(value))

        # 2. Fall back to content parsing
        if not urls:
            urls = set(_URL_PATTERN.findall(doc.content))

        doc.metadata["outgoing_urls"] = sorted(urls)

    logger.info("Normalized links for %d documents", len(docs))
    return docs
