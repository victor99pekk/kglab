"""Composite relation extractor — chains multiple extractors and merges results."""

import logging
from typing import Any

from polygraph._shared import Ontology
from polygraph.kg_build.extract._base import Entity, RelationExtractorMethod, Triple
from polygraph.kg_build.extract.registry import create_relation_method

logger = logging.getLogger(__name__)


class CompositeRelationExtractor(RelationExtractorMethod):
    """Combine multiple relation extraction methods into one.

    Each sub-extractor is instantiated lazily via the registry, so the
    composite works with any registered relation method. Triples from all
    sub-extractors are merged and deduplicated by ``(subject, predicate, object)``.

    Usage::

        composite = CompositeRelationExtractor(
            ontology=ontology,
            methods=["ontology_rules", "structured_llm"],
            method_options={"structured_llm": {"model_name": "deepseek-v4-flash"}},
        )
        triples = composite.extract(text, entities, source_chunk_id="chunk:1")
    """

    def __init__(
        self,
        ontology: Ontology,
        methods: list[str],
        method_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.ontology = ontology
        self.methods = methods
        self._method_options = method_options or {}
        self._extractors: list[RelationExtractorMethod] = []

    def _ensure_extractors(self) -> list[RelationExtractorMethod]:
        """Lazily instantiate sub-extractors on first call."""
        if not self._extractors:
            for method_name in self.methods:
                options = self._method_options.get(method_name, {})
                extractor = create_relation_method(
                    method_name,
                    ontology=self.ontology,
                    **options,
                )
                self._extractors.append(extractor)
        return self._extractors

    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[Triple]:
        all_triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        for extractor in self._ensure_extractors():
            for triple in extractor.extract(text, entities, source_chunk_id):
                key = (triple[0], triple[1], triple[2])
                if key not in seen:
                    seen.add(key)
                    all_triples.append(triple)

        logger.debug(
            "CompositeRelationExtractor: %d unique triples from %d methods",
            len(all_triples),
            len(self._extractors),
        )
        return all_triples
