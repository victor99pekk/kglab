"""Abstract base class for document-to-document relation extraction."""

from abc import ABC, abstractmethod
from typing import Any

from kglab._shared import Document

#: A triple with optional evidence and source info.
#: ``(subject_id, predicate, object_id, evidence, source_chunk_id)``
DocTriple = tuple[str, ...]


class DocumentRelationExtractor(ABC):
    """Contract for methods that extract relations between documents.

    Unlike :class:`~kglab.kg_build.extract._base.RelationExtractorMethod`,
    which operates on one chunk at a time, this interface receives the full
    list of source documents. This enables cross-document relations such as
    hyperlinks, citations, and shared authorship.

    Subclasses must implement :meth:`extract` and return two things:

    * **document_entities** — a list of dicts describing each Document node
      to add to the graph (at minimum ``id``, ``name``, and ``type``).
    * **triples** — a list of ``(subject_id, predicate, object_id, ...)``
      tuples representing document-to-document edges.
    """

    @abstractmethod
    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        """Extract document nodes and document-to-document triples.

        Args:
            documents: The full list of source documents (pre-chunking).

        Returns:
            A ``(doc_entities, triples)`` tuple where *doc_entities* are dicts
            suitable for graph node creation and *triples* are document-to-document
            edge tuples.
        """
        ...
