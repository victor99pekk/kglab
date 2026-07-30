"""Shared helpers for document-to-document relation extraction."""

from typing import Any

from polygraph._shared import Document, entity_id


def make_document_entities(documents: list[Document]) -> list[dict[str, Any]]:
    """Build DOCUMENT entity dicts for every source document."""
    entities: list[dict[str, Any]] = []
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
            }
        )
    return entities


def doc_entity_id(doc: Document) -> str:
    """Stable entity ID for a document node."""
    return entity_id("Document", doc.doc_id)


def make_chunk_entities(chunks: list[Document]) -> list[dict[str, Any]]:
    """Build Chunk entity dicts from preprocessed chunks."""
    entities: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        chunk.metadata.get("parent_doc_id", chunk.source)
        entities.append(
            {
                "id": chunk.doc_id,
                "name": chunk.doc_id,
                "type": "Chunk",
                "source": chunk.source,
                "text": chunk.content,
                "tokenCount": chunk.metadata.get("token_count", 0),
                "index": chunk.metadata.get("chunk_index", i),
            }
        )
    return entities
