"""Structured-JSON LLM English relation extraction method."""

import json
import logging
import os
import re
from typing import Any

from polygraph._shared import Language, Ontology
from polygraph.kg_build.extract._base import Entity, RelationExtractorMethod, Triple
from polygraph.kg_build.extract.relation.en._helpers import find_evidence

logger = logging.getLogger(__name__)


class StructuredLLMRelationExtractor(RelationExtractorMethod):
    """Ask an OpenAI-compatible model for relations between known entities."""

    def __init__(
        self,
        ontology: Ontology,
        language: Language = Language.ENGLISH,
        model_name: str = "deepseek-v4-flash",
        client: Any | None = None,
        max_text_chars: int = 8_000,
    ) -> None:
        self.ontology = ontology
        self.language = language
        self.model_name = model_name
        self._provided_client = client
        self.max_text_chars = max_text_chars
        self._relation_schema = {
            predicate.casefold(): {
                "predicate": predicate,
                "domain": domain,
                "range": range_,
                "symmetric": symmetric,
                "description": ontology.relationship_types[predicate].get("description", ""),
            }
            for domain, range_, predicate, symmetric in ontology.get_relation_patterns()
        }

    def extract(
        self,
        text: str,
        entities: list[Entity],
        source_chunk_id: str = "",
    ) -> list[Triple]:
        if not text.strip() or len(entities) < 2 or not self._relation_schema:
            return []

        client = self._client()

        entity_list = "\n".join(f"- {entity.name} ({entity.label})" for entity in entities)
        relation_list = "\n".join(
            f"- {schema['predicate']}: {schema['domain']} -> {schema['range']}"
            + (f" — {schema['description']}" if schema["description"] else "")
            for schema in self._relation_schema.values()
        )
        model_text = text[: self.max_text_chars]
        prompt = (
            "Given the following text and extracted entities, identify all relationships "
            "between entities. For each relationship, include the exact sentence from the "
            "text that supports it. Use only the allowed predicates below, with the shown "
            "subject and object types. If no supported relationship exists, return []. "
            "Output a JSON list of objects with subject, predicate, object, and evidence "
            f"fields.\n\nText:\n{model_text}\n\nEntities:\n{entity_list}\n\n"
            f"Allowed predicates:\n{relation_list}\n\nReturn ONLY JSON, e.g. "
            '[{"subject":"Alice","predicate":"works_at","object":"Acme Corp",'
            '"evidence":"Alice works at Acme Corp."}]'
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content or "[]"
        raw_triples = self._decode_records(content)
        if raw_triples is None:
            return []

        entity_index: dict[str, Entity] = {}
        for entity in entities:
            entity_index.setdefault(entity.name.casefold().strip(), entity)
            for alias in entity.aliases:
                entity_index.setdefault(alias.casefold().strip(), entity)

        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for triple in raw_triples:
            parsed = self._parse_record(triple)
            if parsed is None:
                continue
            subject_name, predicate, object_name, evidence = parsed
            subject = entity_index.get(subject_name.casefold().strip())
            object_ = entity_index.get(object_name.casefold().strip())
            relation = self._relation_schema.get(
                predicate.casefold().strip().replace(" ", "_")
            )
            if subject is None or object_ is None or subject.id == object_.id or relation is None:
                continue
            if subject.label != relation["domain"] or object_.label != relation["range"]:
                continue

            grounded_evidence = self._ground_evidence(
                model_text,
                subject,
                object_,
                subject_name,
                object_name,
                evidence,
            )
            if not grounded_evidence:
                continue

            key = (subject.id, relation["predicate"], object_.id)
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                (
                    subject.id,
                    relation["predicate"],
                    object_.id,
                    grounded_evidence,
                    source_chunk_id,
                )
            )

        logger.debug("StructuredLLMRelationExtractor: found %d triples", len(triples))
        return triples

    def _client(self):
        if self._provided_client is not None:
            return self._provided_client

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to .env before using structured_llm."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The LLM dependencies are missing. Install with: uv sync --extra llm"
            ) from exc

        self._provided_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        return self._provided_client

    @staticmethod
    def _decode_records(content: str) -> list[object] | None:
        """Decode a JSON list, optionally wrapped in a Markdown fence or object."""
        cleaned = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM relation output")
            return None

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("relations"), list):
            return payload["relations"]
        logger.warning("LLM relation output must be a JSON list")
        return None

    @staticmethod
    def _ground_evidence(
        text: str,
        subject: Entity,
        object_: Entity,
        subject_name: str,
        object_name: str,
        supplied: str,
    ) -> str:
        """Return a source sentence containing both endpoints, or reject it."""
        if supplied and supplied in text:
            supplied_key = supplied.casefold()
            if (
                subject_name.casefold() in supplied_key
                and object_name.casefold() in supplied_key
            ):
                return supplied

        evidence = find_evidence(text, subject_name, object_name)
        if evidence:
            return evidence
        return find_evidence(text, subject.name, object_.name)

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
