"""Storage-agnostic graph writer — one build routine, any backend.

The same KG build logic can target different storage backends by writing
through the ``GraphWriter`` interface:

* ``NetworkXGraphWriter`` — builds an in-memory ``nx.DiGraph`` (a plain
  Python object).
* ``Neo4jGraphBuilder`` (``kglab.kg_export.neo4j.builder``) — streams
  nodes and edges directly into a Neo4j database.

``build_kg_into`` is the single, shared build routine: given chunks,
entities, and triples, it writes documents, chunks, structural edges
(``PART_OF`` / ``NEXT``), and semantic edges through any ``GraphWriter``.
Swap the writer to change where the graph is stored — the pipeline code
stays the same::

    from kglab.kg_build.build import NetworkXGraphWriter, build_kg_into

    # In-memory (python object):
    writer = NetworkXGraphWriter(ontology=ontology)
    build_kg_into(writer, chunks, entities, triples)
    graph = writer.graph  # nx.DiGraph

    # Neo4j (streamed, no in-memory graph) — same call:
    from kglab.kg_export.neo4j.builder import Neo4jGraphBuilder
    build_kg_into(builder, chunks, entities, triples)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from kglab._shared import Document
from kglab.kg_build.build.networkx import GraphBuilder

#: Entity dict keys handled as explicit ``merge_entity`` arguments; everything
#: else is forwarded as generic node properties.
_ENTITY_BASE_FIELDS = {
    "id",
    "name",
    "type",
    "description",
    "aliases",
    "embedding",
    "confidenceScore",
    "importanceScore",
}


class GraphWriter(ABC):
    """Incremental write interface shared by every graph storage backend.

    Implement these methods to make a backend usable by ``build_kg_into``.
    See ``NetworkXGraphWriter`` (in-memory) and ``Neo4jGraphBuilder``
    (streamed to Neo4j) for reference implementations.
    """

    @abstractmethod
    def merge_document(
        self,
        doc_id: str,
        *,
        name: str = "",
        description: str = "",
        source: str = "",
        chunk_count: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a ``:Document`` node."""

    @abstractmethod
    def merge_chunk(
        self,
        chunk_id: str,
        *,
        source: str = "",
        text: str = "",
        token_count: int = 0,
        index: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a ``:Chunk`` node."""

    @abstractmethod
    def merge_entity(
        self,
        entity_id: str,
        *,
        name: str = "",
        entity_type: str = "Entity",
        description: str = "",
        importance_score: float = 0.0,
        confidence_score: float = 1.0,
        embedding: list[float] | None = None,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update an entity node."""

    @abstractmethod
    def merge_edge(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        *,
        evidence_sentence: str = "",
        source_chunk_id: str = "",
        description: str = "",
        weight: int = 1,
    ) -> None:
        """Create or update a relationship between two nodes."""

    @abstractmethod
    def merge_structural_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> None:
        """Create a simple structural edge (e.g. ``PART_OF``, ``NEXT``)."""

    @property
    @abstractmethod
    def stats(self) -> dict[str, int]:
        """Running counters of nodes and edges written during this session."""


class NetworkXGraphWriter(GraphWriter):
    """In-memory ``GraphWriter`` that produces an ``nx.DiGraph``.

    Writes are accumulated and materialised into exactly the same graph the
    batch ``GraphBuilder`` produces, so results are identical whether a
    pipeline builds via the batch API or through this incremental writer.
    """

    def __init__(self, ontology: Any | None = None) -> None:
        self.ontology = ontology
        self._entities: list[dict[str, Any]] = []
        self._triples: list[tuple] = []
        self._graph: Any | None = None
        self._node_count = 0
        self._edge_count = 0

    # ── GraphWriter implementation ───────────────────────────────

    def merge_document(
        self,
        doc_id: str,
        *,
        name: str = "",
        description: str = "",
        source: str = "",
        chunk_count: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self._entities.append(
            {
                "id": doc_id,
                "type": "Document",
                "name": name or doc_id,
                "description": description,
                "source": source,
                "chunk_count": chunk_count,
                **(properties or {}),
            }
        )
        self._node_count += 1

    def merge_chunk(
        self,
        chunk_id: str,
        *,
        source: str = "",
        text: str = "",
        token_count: int = 0,
        index: int = 0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self._entities.append(
            {
                "id": chunk_id,
                "type": "Chunk",
                "name": chunk_id,
                "source": source,
                "text": text,
                "tokenCount": token_count,
                "index": index,
                **(properties or {}),
            }
        )
        self._node_count += 1

    def merge_entity(
        self,
        entity_id: str,
        *,
        name: str = "",
        entity_type: str = "Entity",
        description: str = "",
        importance_score: float = 0.0,
        confidence_score: float = 1.0,
        embedding: list[float] | None = None,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self._entities.append(
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "description": description,
                "importanceScore": importance_score,
                "confidenceScore": confidence_score,
                "embedding": embedding,
                "aliases": aliases or [],
                **(properties or {}),
            }
        )
        self._node_count += 1

    def merge_edge(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        *,
        evidence_sentence: str = "",
        source_chunk_id: str = "",
        description: str = "",
        weight: int = 1,
    ) -> None:
        triple: list = [source_id, predicate, target_id, evidence_sentence, source_chunk_id]
        if description:
            triple.append(description)
        self._triples.append(tuple(triple))
        self._edge_count += 1

    def merge_structural_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> None:
        self._triples.append((source_id, relationship_type, target_id, "", ""))
        self._edge_count += 1

    # ── Access ───────────────────────────────────────────────────

    @property
    def graph(self) -> Any:
        """The built ``nx.DiGraph`` (materialised lazily on first access)."""
        if self._graph is None:
            self._graph = GraphBuilder(ontology=self.ontology).build(self._entities, self._triples)
        return self._graph

    @property
    def stats(self) -> dict[str, int]:
        return {"nodes_written": self._node_count, "edges_written": self._edge_count}


def build_kg_into(
    writer: GraphWriter,
    chunks: list[Document],
    entities: list[dict[str, Any]],
    triples: list[tuple],
) -> dict[str, int]:
    """Write a complete KG through any ``GraphWriter`` (shared build routine).

    This is the single build implementation used regardless of where the
    graph is stored.  It writes:

    * document and chunk nodes plus ``PART_OF`` / ``NEXT`` structural edges
    * entity nodes
    * semantic triples as edges

    Args:
        writer: Any backend implementing ``GraphWriter`` (e.g.
            ``NetworkXGraphWriter`` for an in-memory graph, or
            ``Neo4jGraphBuilder`` to stream into Neo4j).
        chunks: Chunked documents (each carries ``parent_doc_id`` metadata).
        entities: Resolved entity dicts (``id``, ``name``, ``type``, ...).
        triples: ``(subject, predicate, object, evidence, chunk_id, ...)``
            tuples.

    Returns:
        The writer's ``stats`` dict.
    """
    # ── Structure: documents, chunks, PART_OF / NEXT edges ───────
    doc_ids = {c.metadata.get("parent_doc_id", c.doc_id) for c in chunks}
    for doc_id in doc_ids:
        if doc_id:
            writer.merge_document(doc_id, name=doc_id)

    for i, chunk in enumerate(chunks):
        writer.merge_chunk(
            chunk.doc_id,
            source=chunk.source,
            text=chunk.content,
            index=i,
        )
        parent = chunk.metadata.get("parent_doc_id", "")
        if parent:
            writer.merge_structural_edge(chunk.doc_id, parent, "PART_OF")

    for i in range(len(chunks) - 1):
        prev, nxt = chunks[i], chunks[i + 1]
        parent_prev = prev.metadata.get("parent_doc_id") or prev.source
        parent_next = nxt.metadata.get("parent_doc_id") or nxt.source
        if parent_prev and parent_prev == parent_next:
            writer.merge_structural_edge(prev.doc_id, nxt.doc_id, "NEXT")

    # ── Entities ─────────────────────────────────────────────────
    for entity in entities:
        writer.merge_entity(
            entity["id"],
            name=entity.get("name", ""),
            entity_type=entity.get("type", "Entity"),
            description=entity.get("description", ""),
            aliases=entity.get("aliases"),
            embedding=entity.get("embedding"),
            confidence_score=entity.get("confidenceScore", 1.0),
            importance_score=entity.get("importanceScore", 0.0),
            # Any remaining entity fields (source_chunk_ids, url, title,
            # upload_date, ...) are carried through as generic properties so
            # in-memory graphs keep the full attribute set.
            properties={
                key: value for key, value in entity.items() if key not in _ENTITY_BASE_FIELDS
            },
        )

    # ── Semantic triples ─────────────────────────────────────────
    for triple in triples:
        writer.merge_edge(
            triple[0],
            triple[2],
            triple[1],
            evidence_sentence=triple[3] if len(triple) > 3 else "",
            source_chunk_id=triple[4] if len(triple) > 4 else "",
        )

    return writer.stats


__all__ = ["GraphWriter", "NetworkXGraphWriter", "build_kg_into"]
