"""Relation extraction between entities."""

import logging
import re

from polygraph._shared import Language, Ontology
from polygraph.kg_build.extract.entities import Entity

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extracts (subject, predicate, object) triples from text.

    Requires an ``Ontology`` that defines entity types and relationship
    types (with optional domain/range and symmetric flags).  The ontology
    is loaded from a YAML file via ``Ontology.from_yaml(path)``.
    """

    def __init__(
        self,
        ontology: Ontology,
        language: Language = Language.ENGLISH,
        use_llm: bool = False,
        model_name: str = "deepseek-v4-flash",
    ) -> None:
        self.ontology = ontology
        self.language = language
        self.use_llm = use_llm
        self.model_name = model_name

        # Derive typed and generic patterns from the ontology.
        raw = ontology.get_relation_patterns()
        self._typed_patterns: list[tuple[str, str, str]] = []
        self._generic_predicates: list[str] = []
        self._symmetric_predicates: set[str] = set()

        for domain, range_, predicate, symmetric in raw:
            if domain and range_:
                self._typed_patterns.append((domain, range_, predicate))
            else:
                self._generic_predicates.append(predicate)
            if symmetric:
                self._symmetric_predicates.add(predicate)

    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[tuple[str, str, str, str, str]]:
        """Extract relation triples from text given extracted entities.

        Returns: (subject_id, predicate, object_id, evidence_sentence, source_chunk_id).
        """
        if self.use_llm:
            return self._llm_extract(text, entities, source_chunk_id)
        return self._rule_based_extract(text, entities, source_chunk_id)

    def _rule_based_extract(
        self, text: str, entities: list[Entity], source_chunk_id: str = ""
    ) -> list[tuple[str, str, str, str, str]]:
        """Rule-based relation extraction using entity co-occurrence and heuristics."""
        triples: list[tuple[str, str, str, str, str]] = []
        # Use entity co-occurrence within the same sentence
        sentences = self._sentences(text)

        for sentence in sentences:
            sent_lower = sentence.lower()
            # Find entities that appear in this sentence
            present = [e for e in entities if e.name.lower() in sent_lower]
            if len(present) < 2:
                continue

            # Try pattern matching
            # Each unordered pair is considered once. The old ordered-pair
            # loop emitted an incorrect reverse edge for every relationship.
            for first_index, first in enumerate(present):
                for second in present[first_index + 1 :]:
                    relation = self._infer_relation(first, second)
                    if relation:
                        subject, predicate, object_ = relation
                        triples.append(
                            (subject.id, predicate, object_.id, sentence, source_chunk_id)
                        )

        logger.debug(f"RuleBasedRelationExtractor: found {len(triples)} triples")
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

    def _infer_relation(
        self,
        e1: Entity,
        e2: Entity,
    ) -> tuple[Entity, str, Entity] | None:
        """Infer one correctly oriented relation for an unordered entity pair.

        Patterns are derived from the ontology passed at construction time.
        """
        # 1. Try typed patterns (domain / range match).
        for head_label, dep_label, predicate in self._typed_patterns:
            if e1.label == head_label and e2.label == dep_label:
                return self._canonicalize_symmetric(e1, predicate, e2)
            if e1.label == dep_label and e2.label == head_label:
                return self._canonicalize_symmetric(e2, predicate, e1)

        # 2. Try generic predicates (no domain/range constraints).
        for predicate in self._generic_predicates:
            return self._canonicalize_symmetric(e1, predicate, e2)

        # 3. Fallback: same-label pairs get "related_to"-style treatment.
        if e1.label == e2.label and self._generic_predicates:
            return self._canonicalize_symmetric(e1, self._generic_predicates[0], e2)
        if self._generic_predicates:
            return self._canonicalize_symmetric(e1, self._generic_predicates[-1], e2)

        return None

    def _canonicalize_symmetric(
        self,
        subject: Entity,
        predicate: str,
        object_: Entity,
    ) -> tuple[Entity, str, Entity]:
        """Ensure canonical orientation for symmetric predicates."""
        if predicate in self._symmetric_predicates and object_.id < subject.id:
            return object_, predicate, subject
        return subject, predicate, object_

    def _llm_extract(
        self, text: str, entities: list[Entity], source_chunk_id: str = ""
    ) -> list[tuple[str, str, str, str, str]]:
        """LLM-based relation extraction using DeepSeek (OpenAI-compatible API)."""
        import os

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — falling back to rule-based extraction")
            return self._rule_based_extract(text, entities, source_chunk_id)

        entity_list = "\n".join(f"- {e.name} ({e.label})" for e in entities)

        prompt = (
            f"Given the following text and extracted entities, identify all relationships "
            f"between entities. For each relationship, include the exact sentence from the "
            f"text that supports it. Output a JSON list of objects with subject, predicate, "
            f"object, and evidence fields.\n\n"
            f"Text:\n{text[:2000]}\n\n"
            f"Entities:\n{entity_list}\n\n"
            f"Return ONLY JSON, e.g. "
            f'[{"{"}"subject":"Alice","predicate":"works_at",'
            f'"object":"Acme Corp","evidence":"Alice works at Acme Corp."{"}"}]'
        )

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "[]"

        import json

        try:
            raw_triples = json.loads(content)
            entity_ids = {entity.name.casefold(): entity.id for entity in entities}
            triples = []
            for triple in raw_triples:
                if isinstance(triple, dict):
                    subject_name = str(triple.get("subject", ""))
                    predicate = str(triple.get("predicate", ""))
                    object_name = str(triple.get("object", ""))
                    evidence = str(triple.get("evidence", "")).strip()
                elif isinstance(triple, list) and len(triple) >= 3:
                    subject_name, predicate, object_name = map(str, triple[:3])
                    evidence = str(triple[3]).strip() if len(triple) > 3 else ""
                else:
                    continue
                subject_id = entity_ids.get(subject_name.casefold())
                object_id = entity_ids.get(object_name.casefold())
                if subject_id and object_id:
                    if not evidence or evidence not in text:
                        evidence = self._find_evidence(text, subject_name, object_name)
                    triples.append((subject_id, predicate, object_id, evidence, source_chunk_id))
            logger.debug(f"LLM extraction: found {len(triples)} triples")
            return triples
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM relation output")
            return []

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split text into evidence-sized sentences while retaining punctuation."""
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if part.strip()]

    @classmethod
    def _find_evidence(cls, text: str, subject: str, object_: str) -> str:
        """Find the sentence containing both relation endpoints."""
        subject_key = subject.casefold()
        object_key = object_.casefold()
        for sentence in cls._sentences(text):
            sentence_key = sentence.casefold()
            if subject_key in sentence_key and object_key in sentence_key:
                return sentence
        return ""
