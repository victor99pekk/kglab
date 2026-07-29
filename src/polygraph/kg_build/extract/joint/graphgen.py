"""Paper-faithful GraphGen KG extraction using DeepSeek as the synthesizer."""

from __future__ import annotations

import html
import logging
import os
import re
from collections import defaultdict
from typing import Any

from polygraph._shared import Language, Ontology
from polygraph.kg_build.extract._base import Entity, JointExtractor

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

FIGURE_8_TEMPLATE = """You are an NLP expert, skilled at analyzing text to extract named entities and their relationships.

-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {output_language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, using the same language as the input text. If English, capitalize the name.
- entity_type: One of the following types: [{entity_types}]
- entity_summary: Comprehensive summary of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_summary: explanation as to why you think the source entity and target entity are related to each other
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {output_language} as a single list of all entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

################
-Examples-
################
-Example 1-
Text:
In the second century of the Christian Era, the empire of Rome comprehended the fairest part of the earth, and the most civilized portion of mankind. The frontiers of that extensive monarchy were guarded by ancient renown and disciplined valor. The gentle but powerful influence of laws and manners had gradually cemented the union of the provinces. Their peaceful inhabitants enjoyed and abused the advantages of wealth and luxury. The image of a free constitution was preserved with decent reverence: the Roman senate appeared to possess the sovereign authority, and devolved on the emperors all the executive powers of government. During a happy period of more than fourscore years, the public administration was conducted by the virtue and abilities of Nerva, Trajan, Hadrian, and the two Antonines.
################
Output:
("entity"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"organization"{tuple_delimiter}"The dominant empire of the second century CE, encompassing the most developed regions of the known world."){record_delimiter}
("entity"{tuple_delimiter}"Second Century CE"{tuple_delimiter}"date"{tuple_delimiter}"Time period of the Christian Era when the Roman Empire was at its height."){record_delimiter}
("entity"{tuple_delimiter}"Rome"{tuple_delimiter}"location"{tuple_delimiter}"The capital and heart of the Roman Empire."){record_delimiter}
("entity"{tuple_delimiter}"Roman Senate"{tuple_delimiter}"organization"{tuple_delimiter}"Legislative body that appeared to hold sovereign authority in Rome."){record_delimiter}
("entity"{tuple_delimiter}"Nerva"{tuple_delimiter}"person"{tuple_delimiter}"Roman emperor who contributed to the public administration during a prosperous period."){record_delimiter}
("entity"{tuple_delimiter}"Trajan"{tuple_delimiter}"person"{tuple_delimiter}"Roman emperor known for his virtue and administrative abilities."){record_delimiter}
("entity"{tuple_delimiter}"Hadrian"{tuple_delimiter}"person"{tuple_delimiter}"Roman emperor who governed during the empire's peaceful period."){record_delimiter}
("entity"{tuple_delimiter}"Antonines"{tuple_delimiter}"person"{tuple_delimiter}"Two Roman emperors who ruled during a period of prosperity and good governance."){record_delimiter}
("entity"{tuple_delimiter}"Roman Law"{tuple_delimiter}"concept"{tuple_delimiter}"System of laws and manners that unified the provinces of the Roman Empire."){record_delimiter}
("relationship"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"Roman Law"{tuple_delimiter}"The empire was unified and maintained through the influence of its laws and customs."){record_delimiter}
("relationship"{tuple_delimiter}"Roman Senate"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"The Senate appeared to possess sovereign authority while delegating executive powers to emperors."){record_delimiter}
("relationship"{tuple_delimiter}"Nerva"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"Nerva was one of the emperors who contributed to the empire's successful administration."){record_delimiter}
("relationship"{tuple_delimiter}"Trajan"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"Trajan was one of the emperors who governed during the empire's prosperous period."){record_delimiter}
("relationship"{tuple_delimiter}"Hadrian"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"Hadrian was one of the emperors who managed the empire's administration effectively."){record_delimiter}
("relationship"{tuple_delimiter}"Antonines"{tuple_delimiter}"Roman Empire"{tuple_delimiter}"The Antonines were emperors who helped maintain the empire's prosperity through their governance."){record_delimiter}
("content_keywords"{tuple_delimiter}"Roman governance, imperial prosperity, law and order, civilized society"){completion_delimiter}

-Example 2-
Text:
Overall, the analysis of the OsDT11 sequence demonstrated that this protein belongs to the CRP family. Since OsDT11 is predicted to be a secreted protein, the subcellular localization of OsDT11 was determined by fusing the OsDT11 ORF to RFP in a p35S::RFP vector by in vivo protein targeting in NB epidermal cells by performing an Agrobacterium tumefaciens-mediated transient assay. After incubation for 48 h, the RFP signals were mainly detected in the cell-wall of OsDT11-RFP transformed cells, while the control cells (transformed with the RFP construct) displayed ubiquitous RFP signals, demonstrating that OsDT11 is a secreted signal peptide. Moreover, when the infiltrated leaf sections were plasmolyzed, the OsDT11-RFP fusion proteins were located on the cell wall.
################
Output:
("entity"{tuple_delimiter}"OsDT11"{tuple_delimiter}"gene"{tuple_delimiter}"A protein sequence belonging to the CRP family, demonstrated to be a secreted signal peptide that localizes to cell walls."){record_delimiter}
("entity"{tuple_delimiter}"CRP family"{tuple_delimiter}"science"{tuple_delimiter}"A protein family to which OsDT11 belongs, characterized by specific structural and functional properties."){record_delimiter}
("entity"{tuple_delimiter}"RFP"{tuple_delimiter}"technology"{tuple_delimiter}"Red Fluorescent Protein, used as a fusion marker to track protein localization in cells."){record_delimiter}
("entity"{tuple_delimiter}"p35S::RFP vector"{tuple_delimiter}"technology"{tuple_delimiter}"A genetic construct used for protein expression and visualization studies, containing the 35S promoter and RFP marker."){record_delimiter}
("entity"{tuple_delimiter}"NB epidermal cells"{tuple_delimiter}"nature"{tuple_delimiter}"Plant epidermal cells used as the experimental system for protein localization studies."){record_delimiter}
("entity"{tuple_delimiter}"Agrobacterium tumefaciens"{tuple_delimiter}"nature"{tuple_delimiter}"A bacteria species used for transferring genetic material into plant cells in laboratory experiments."){record_delimiter}
("relationship"{tuple_delimiter}"OsDT11"{tuple_delimiter}"CRP family"{tuple_delimiter}"OsDT11 is identified as a member of the CRP family through sequence analysis."){record_delimiter}
("relationship"{tuple_delimiter}"OsDT11"{tuple_delimiter}"RFP"{tuple_delimiter}"OsDT11 was fused to RFP to study its cellular localization."){record_delimiter}
("relationship"{tuple_delimiter}"Agrobacterium tumefaciens"{tuple_delimiter}"NB epidermal cells"{tuple_delimiter}"Agrobacterium tumefaciens was used to transfer genetic material into NB epidermal cells through a transient assay."){record_delimiter}
("relationship"{tuple_delimiter}"OsDT11"{tuple_delimiter}"NB epidermal cells"{tuple_delimiter}"OsDT11's subcellular localization was studied in NB epidermal cells, showing cell wall targeting."){record_delimiter}
("content_keywords"{tuple_delimiter}"protein localization, gene expression, cellular biology, molecular techniques"){completion_delimiter}

################
-Real Data-
################
Entity_types: {entity_types}
Text: {input_text}
################
Output:
"""

CONTINUE_PROMPT = """MANY entities and relationships were missed in the last extraction. Add them below using the same format:"""

IF_LOOP_PROMPT = """It appears some entities and relationships may have still been missed. Answer YES | NO if there are still entities and relationships that need to be added."""

FIGURE_9_SUMMARIZATION_TEMPLATE = """You are an NLP expert responsible for generating a comprehensive summary of the data provided below.
Given one entity or relationship and a list of descriptions all related to that same entity or relationship, combine them into one comprehensive description. Include information from every description. If descriptions contradict each other, resolve the contradictions and provide one coherent summary. Write in the third person and include the entity names for full context.
Use {output_language} as output language.

#######
-Data-
Entity or relationship: {name}
Description List: {description_list}
#######
Output:
"""


class GraphGenExtractor(JointExtractor):
    """Extract and aggregate the descriptive graph used by GraphGen.

    Figure 8 does not predict a predicate taxonomy. Each knowledge edge is an
    ordered entity pair plus a natural-language description. ``RELATION`` is
    only the neutral carrier label required by our tuple/Neo4j representation.
    """

    prompt_version = "graphgen-figure8-figure9-v2"

    def __init__(
        self,
        ontology: Ontology,
        language: Language = Language.ENGLISH,
        model_name: str = "deepseek-v4-flash",
        client: Any | None = None,
        max_retries: int = 3,
        max_gleanings: int = 3,
    ) -> None:
        self.ontology = ontology
        self.language = language
        self.model_name = model_name
        entity_names = ontology.get_entity_type_names()
        if not entity_names:
            raise ValueError(
                "GraphGenExtractor requires an ontology with at least one entity type. "
                "Load one via Ontology.from_yaml(path)."
            )
        self.entity_types = tuple(name.upper() for name in entity_names)
        self._provided_client = client
        self.max_retries = max_retries
        self.max_gleanings = max_gleanings

    @property
    def output_language(self) -> str:
        return "English"

    def extract(
        self,
        text: str,
        source_chunk_id: str = "",
    ) -> tuple[list[Entity], list[tuple[str, ...]]]:
        """Run Figure 8 extraction and the reference iterative gleaning loop."""
        if not text.strip():
            return [], []

        prompt_template = FIGURE_8_TEMPLATE
        prompt = prompt_template.format(
            output_language=self.output_language,
            entity_types=", ".join(entity_type.lower() for entity_type in self.entity_types),
            tuple_delimiter=TUPLE_DELIMITER,
            record_delimiter=RECORD_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            input_text=text,
        )
        initial = self._generate([{"role": "user", "content": prompt}])
        final_result = initial
        history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial},
        ]

        for _ in range(self.max_gleanings):
            if_loop = self._generate(
                [
                    *history,
                    {
                        "role": "user",
                        "content": IF_LOOP_PROMPT,
                    },
                ],
                max_tokens=64,
            )
            if if_loop.strip().strip('"').strip("'").casefold() != "yes":
                break

            glean = self._generate(
                [
                    *history,
                    {
                        "role": "user",
                        "content": CONTINUE_PROMPT,
                    },
                ]
            )
            final_result += f"{RECORD_DELIMITER}{glean}"
            history.extend(
                [
                    {
                        "role": "user",
                        "content": CONTINUE_PROMPT,
                    },
                    {"role": "assistant", "content": glean},
                ]
            )

        return self._parse(final_result, source_chunk_id)

    def aggregate_descriptions(
        self,
        resolved_entities: list[dict[str, Any]],
        original_entities: list[dict[str, Any]],
        entity_id_map: dict[str, str],
        triples: list[tuple[str, ...]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, ...]]]:
        """Apply Figure 9 to repeated entity and relationship descriptions."""
        entity_descriptions: dict[str, list[str]] = defaultdict(list)
        for entity in original_entities:
            entity_id = str(entity.get("id", ""))
            canonical_id = entity_id_map.get(entity_id, entity_id)
            description = str(entity.get("description", "")).strip()
            if canonical_id and description:
                entity_descriptions[canonical_id].append(description)

        for entity in resolved_entities:
            descriptions = entity_descriptions.get(str(entity.get("id", "")), [])
            if descriptions:
                entity["description"] = self._merge_descriptions(
                    str(entity.get("name", "entity")), descriptions
                )

        relation_descriptions: dict[tuple[str, str], list[str]] = defaultdict(list)
        for triple in triples:
            if len(triple) > 5 and triple[1] == "RELATION" and triple[5]:
                relation_descriptions[(triple[0], triple[2])].append(str(triple[5]))

        merged_relations = {
            key: self._merge_descriptions(f"({key[0]}, {key[1]})", descriptions)
            for key, descriptions in relation_descriptions.items()
        }
        aggregated_triples = [
            (
                *triple[:5],
                merged_relations[(triple[0], triple[2])],
            )
            if triple[1] == "RELATION" and (triple[0], triple[2]) in merged_relations
            else triple
            for triple in triples
        ]
        return resolved_entities, aggregated_triples

    def _client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env before using --llm.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The LLM dependencies are missing. Install with: "
                "/Users/armon/.local/bin/uv sync --extra llm"
            ) from exc

        self._provided_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        return self._provided_client

    def _generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client().chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("DeepSeek returned an empty response")
                return content.strip()
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "GraphGen request attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

        raise RuntimeError(
            f"DeepSeek GraphGen request failed after {self.max_retries} attempts"
        ) from last_error

    def _parse(
        self,
        content: str,
        source_chunk_id: str,
    ) -> tuple[list[Entity], list[tuple[str, ...]]]:
        records = re.split(
            f"{re.escape(RECORD_DELIMITER)}|{re.escape(COMPLETION_DELIMITER)}",
            content,
        )
        entity_records: list[tuple[str, str, str]] = []
        relationship_records: list[tuple[str, str, str]] = []
        content_keywords: list[str] = []

        for record in records:
            match = re.search(r"\((.*)\)", record.strip(), flags=re.DOTALL)
            if not match:
                continue
            fields = [self._clean(field) for field in match.group(1).split(TUPLE_DELIMITER)]
            if len(fields) < 2:
                continue
            record_type = fields[0].casefold()
            if record_type == "entity" and len(fields) >= 4:
                name, entity_type, description = fields[1], fields[2].upper(), fields[3]
                if name and entity_type in self.entity_types:
                    entity_records.append((name, entity_type, description))
            elif record_type == "relationship" and len(fields) >= 4:
                relationship_records.append((fields[1], fields[2], fields[3]))
            elif record_type == "content_keywords":
                content_keywords.extend(
                    keyword.strip() for keyword in fields[1].split(",") if keyword.strip()
                )

        entities = [
            Entity(
                name=name,
                label=entity_type,
                mentions=[name],
                description=description,
                source=source_chunk_id,
            )
            for name, entity_type, description in entity_records
        ]
        entities_by_name: dict[str, Entity] = {}
        for entity in entities:
            entities_by_name.setdefault(entity.name.casefold(), entity)

        relationships: list[tuple[str, ...]] = []
        for source_name, target_name, description in relationship_records:
            source = entities_by_name.get(source_name.casefold())
            target = entities_by_name.get(target_name.casefold())
            if source is None or target is None or source.id == target.id:
                logger.warning(
                    "Discarding GraphGen relationship with a missing/identical endpoint: %s -> %s",
                    source_name,
                    target_name,
                )
                continue
            relationships.append(
                (
                    source.id,
                    "RELATION",
                    target.id,
                    "",
                    source_chunk_id,
                    description,
                )
            )

        logger.debug(
            "GraphGenExtractor: %d entities, %d relationships, keywords=%s",
            len(entities),
            len(relationships),
            content_keywords,
        )
        return entities, relationships

    def _merge_descriptions(self, name: str, descriptions: list[str]) -> str:
        unique = sorted(
            {description.strip() for description in descriptions if description.strip()}
        )
        if not unique:
            return ""
        if len(unique) == 1:
            return unique[0]

        summary_template = FIGURE_9_SUMMARIZATION_TEMPLATE
        prompt = summary_template.format(
            output_language=self.output_language,
            name=name,
            description_list=unique,
        )
        return self._generate([{"role": "user", "content": prompt}])

    @staticmethod
    def _clean(value: str) -> str:
        value = html.unescape(value.strip())
        value = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", value)
        return value.strip().strip('"').strip("'").strip()
