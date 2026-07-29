"""Structured-JSON LLM relation extraction method."""

import json
import logging
import os
from typing import Any

from polygraph._shared import Language, Ontology
from polygraph.kg_build.extract._base import Entity, RelationExtractorMethod, Triple
from polygraph.kg_build.extract.relation._helpers import find_evidence
from polygraph.kg_build.extract.relation.ontology_rules import (
    OntologyRuleRelationExtractor,
)

logger = logging.getLogger(__name__)


class StructuredLLMRelationExtractor(RelationExtractorMethod):
    """Ask an OpenAI-compatible model for relations between known entities."""

    def __init__(
        self,
        ontology: Ontology,
        language: Language = Language.ENGLISH,
        model_name: str = "deepseek-v4-flash",
        client: Any | None = None,
    ) -> None:
        self.ontology = ontology
        self.language = language
        self.model_name = model_name
        self._provided_client = client
        self._fallback = OntologyRuleRelationExtractor(ontology, language)

    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[Triple]:
        try:
            client = self._client()
        except ImportError:
            logger.warning("openai not installed; falling back to ontology rules")
            return self._fallback.extract(text, entities, source_chunk_id)

        entity_list = "\n".join(f"- {entity.name} ({entity.label})" for entity in entities)
        prompt = (
            "Given the following text and extracted entities, identify all relationships "
            "between entities. For each relationship, include the exact sentence from the "
            "text that supports it. Output a JSON list of objects with subject, predicate, "
            f"object, and evidence fields.\n\nText:\n{text[:2000]}\n\n"
            f"Entities:\n{entity_list}\n\nReturn ONLY JSON, e.g. "
            '[{"subject":"Alice","predicate":"works_at","object":"Acme Corp",'
            '"evidence":"Alice works at Acme Corp."}]'
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "[]"

        try:
            raw_triples = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM relation output")
            return []

        entity_ids = {entity.name.casefold(): entity.id for entity in entities}
        triples: list[Triple] = []
        for triple in raw_triples:
            parsed = self._parse_record(triple)
            if parsed is None:
                continue
            subject_name, predicate, object_name, evidence = parsed
            subject_id = entity_ids.get(subject_name.casefold())
            object_id = entity_ids.get(object_name.casefold())
            if subject_id and object_id:
                if not evidence or evidence not in text:
                    evidence = find_evidence(text, subject_name, object_name)
                triples.append((subject_id, predicate, object_id, evidence, source_chunk_id))

        logger.debug("StructuredLLMRelationExtractor: found %d triples", len(triples))
        return triples

    def _client(self):
        if self._provided_client is not None:
            return self._provided_client
        from openai import OpenAI

        self._provided_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        return self._provided_client

    @staticmethod
    def _parse_record(record: object) -> tuple[str, str, str, str] | None:
        if isinstance(record, dict):
            return (
                str(record.get("subject", "")),
                str(record.get("predicate", "")),
                str(record.get("object", "")),
                str(record.get("evidence", "")).strip(),
            )
        if isinstance(record, list) and len(record) >= 3:
            evidence = str(record[3]).strip() if len(record) > 3 else ""
            return str(record[0]), str(record[1]), str(record[2]), evidence
        return None
