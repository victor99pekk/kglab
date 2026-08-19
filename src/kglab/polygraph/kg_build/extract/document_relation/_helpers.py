"""Shared helpers for document-to-document relation extraction."""

from datetime import UTC, datetime
from typing import Any

from kglab._shared import Document, entity_id


def _now_utc() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def make_document_entities(documents: list[Document]) -> list[dict[str, Any]]:
    """Build DOCUMENT entity dicts for every source document."""
    entities: list[dict[str, Any]] = []
    uploaded = _now_utc()
    for doc in documents:
        doc_url = doc.metadata.get("url", doc.doc_id)
        title = doc.metadata.get("title", doc_url)
        entities.append(
            {
                "id": doc_entity_id(doc),
                "name": title,
                "type": "Document",
                "description": doc.metadata.get("description", ""),
                "url": doc_url,
                "source": doc.source,
                "upload_date": uploaded,
            }
        )
    return entities


def doc_entity_id(doc: Document) -> str:
    """Stable entity ID for a document node."""
    return entity_id("Document", doc.doc_id)


def make_chunk_entities(chunks: list[Document]) -> list[dict[str, Any]]:
    """Build Chunk entity dicts from preprocessed chunks."""
    entities: list[dict[str, Any]] = []
    uploaded = _now_utc()
    for i, chunk in enumerate(chunks):
        entities.append(
            {
                "id": chunk.doc_id,
                "name": chunk.doc_id,
                "type": "Chunk",
                "source": chunk.source,
                "text": chunk.content,
                "tokenCount": chunk.metadata.get("token_count", 0),
                "index": chunk.metadata.get("chunk_index", i),
                "upload_date": uploaded,
            }
        )
    return entities
