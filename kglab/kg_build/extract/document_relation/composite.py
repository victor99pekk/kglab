"""Composite document relation extractor — chains multiple extractors and merges results."""

import logging
from typing import Any

from kglab._shared import Document
from kglab.kg_build.extract.document_relation._base import (
    DocTriple,
    DocumentRelationExtractor,
)
from kglab.kg_build.extract.document_relation.registry import create_doc_relation_method

logger = logging.getLogger(__name__)


class CompositeDocRelationExtractor(DocumentRelationExtractor):
    """Combine multiple document relation extraction methods into one.

    Each sub-extractor is instantiated lazily via the registry. Document
    entities are merged (first writer wins for duplicate IDs), and triples
    are deduplicated by ``(subject, predicate, object)``.

    Usage::

        composite = CompositeDocRelationExtractor(
            methods=["hyperlink", "shared_authors", "series"],
        )
        doc_entities, triples = composite.extract(documents)
    """

    def __init__(
        self,
        methods: list[str],
        method_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.methods = methods
        self._method_options = method_options or {}
        self._extractors: list[DocumentRelationExtractor] = []

    def _ensure_extractors(self) -> list[DocumentRelationExtractor]:
        """Lazily instantiate sub-extractors on first call."""
        if not self._extractors:
            for method_name in self.methods:
                options = self._method_options.get(method_name, {})
                extractor = create_doc_relation_method(method_name, **options)
                self._extractors.append(extractor)
        return self._extractors

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        all_entities: dict[str, dict[str, Any]] = {}
        all_triples: list[DocTriple] = []
        seen_triples: set[tuple[str, str, str]] = set()

        for extractor in self._ensure_extractors():
            doc_entities, triples = extractor.extract(documents)
            for entity in doc_entities:
                eid = entity["id"]
                if eid not in all_entities:
                    all_entities[eid] = entity
            for triple in triples:
                key = (triple[0], triple[1], triple[2])
                if key not in seen_triples:
                    seen_triples.add(key)
                    all_triples.append(triple)

        logger.debug(
            "CompositeDocRelationExtractor: %d unique entities, %d triples from %d methods",
            len(all_entities),
            len(all_triples),
            len(self._extractors),
        )
        return list(all_entities.values()), all_triples
