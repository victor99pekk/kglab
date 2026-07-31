"""Ontology-guided, rule-based English relation extraction method."""

import logging

from polygraph._shared import Language, Ontology
from polygraph.kg_build.extract._base import Entity, RelationExtractorMethod, Triple
from polygraph.kg_build.extract.relation.en._helpers import sentences

logger = logging.getLogger(__name__)


class OntologyRuleRelationExtractor(RelationExtractorMethod):
    """Infer relations from sentence co-occurrence and ontology domain/range pairs."""

    def __init__(
        self,
        ontology: Ontology,
        language: Language = Language.ENGLISH,
    ) -> None:
        self.ontology = ontology
        self.language = language
        self._typed_patterns: list[tuple[str, str, str]] = []
        self._symmetric_predicates: set[str] = set()

        for domain, range_, predicate, symmetric in ontology.get_relation_patterns():
            self._typed_patterns.append((domain, range_, predicate))
            if symmetric:
                self._symmetric_predicates.add(predicate)

    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[Triple]:
        triples: list[Triple] = []
        for sentence in sentences(text):
            sentence_key = sentence.casefold()
            present = [entity for entity in entities if entity.name.casefold() in sentence_key]
            if len(present) < 2:
                continue

            for first_index, first in enumerate(present):
                for second in present[first_index + 1 :]:
                    relation = self._infer_relation(first, second)
                    if relation:
                        subject, predicate, object_ = relation
                        triples.append(
                            (subject.id, predicate, object_.id, sentence, source_chunk_id)
                        )

        logger.debug("OntologyRuleRelationExtractor: found %d triples", len(triples))
        return triples

    def _infer_predicate(
        self,
        sentence: str,
        e1: Entity,
        e2: Entity,
        entity_map: dict[str, Entity],
    ) -> str | None:
        """Backward-compatible predicate-only view of relation inference."""
        relation = self._infer_relation(e1, e2)
        return relation[1] if relation else None

    def _infer_relation(self, e1: Entity, e2: Entity) -> tuple[Entity, str, Entity] | None:
        for head_label, dependent_label, predicate in self._typed_patterns:
            if e1.label == head_label and e2.label == dependent_label:
                return self._canonicalize_symmetric(e1, predicate, e2)
            if e1.label == dependent_label and e2.label == head_label:
                return self._canonicalize_symmetric(e2, predicate, e1)

        return None

    def _canonicalize_symmetric(
        self,
        subject: Entity,
        predicate: str,
        object_: Entity,
    ) -> tuple[Entity, str, Entity]:
        if predicate in self._symmetric_predicates and object_.id < subject.id:
            return object_, predicate, subject
        return subject, predicate, object_
