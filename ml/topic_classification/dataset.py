"""Data contracts for multilabel Wikipedia topic classification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TopicDefinition:
    """One stable topic in the configured taxonomy."""

    topic_id: str
    name: str
    domain_id: str
    domain_name: str
    description: str = ""


@dataclass(frozen=True)
class TopicTaxonomy:
    """Validated fixed topic vocabulary and its domain hierarchy."""

    version: str
    topics: dict[str, TopicDefinition]
    domains: dict[str, tuple[str, ...]]
    unknown_label: str = "unknown_topic"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TopicTaxonomy:
        """Load and validate a taxonomy YAML file."""
        taxonomy_path = Path(path)
        payload = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Taxonomy must be a mapping: {taxonomy_path}")

        topics: dict[str, TopicDefinition] = {}
        domains: dict[str, tuple[str, ...]] = {}
        for domain in payload.get("domains", []):
            domain_id = str(domain["id"])
            domain_name = str(domain["name"])
            topic_ids: list[str] = []
            for topic in domain.get("topics", []):
                topic_id = str(topic["id"])
                if topic_id in topics:
                    raise ValueError(f"Duplicate topic ID: {topic_id}")
                topics[topic_id] = TopicDefinition(
                    topic_id=topic_id,
                    name=str(topic["name"]),
                    domain_id=domain_id,
                    domain_name=domain_name,
                    description=str(topic.get("description", "")),
                )
                topic_ids.append(topic_id)
            domains[domain_id] = tuple(topic_ids)

        expected_count = int(payload.get("label_count", len(topics)))
        if len(topics) != expected_count:
            raise ValueError(f"Taxonomy declares {expected_count} labels but defines {len(topics)}")
        return cls(
            version=str(payload.get("version", "")),
            topics=topics,
            domains=domains,
            unknown_label=str(payload.get("unknown_label", "unknown_topic")),
        )


@dataclass(frozen=True)
class ArticleTopicLabel:
    """Topic labels and audit metadata for one article."""

    article_id: str
    topic_ids: tuple[str, ...]
    confidence: float = 1.0
    provenance: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ArticleExample:
    """Article text plus graph-ready relations and multilabel targets."""

    article_id: str
    title: str
    text: str
    topic_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    linked_article_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class TopicClassificationDataset:
    """English Wikipedia articles aligned to the fixed topic taxonomy."""

    def __init__(
        self,
        examples: list[ArticleExample],
        taxonomy: TopicTaxonomy,
    ) -> None:
        self.examples = examples
        self.taxonomy = taxonomy
        self._validate()

    @classmethod
    def from_files(
        cls,
        articles_path: str | Path,
        labels_path: str | Path,
        taxonomy_path: str | Path,
        *,
        require_labels: bool = True,
    ) -> TopicClassificationDataset:
        """Load source articles and separate auditable label records."""
        taxonomy = TopicTaxonomy.from_yaml(taxonomy_path)
        labels = _load_labels(Path(labels_path))
        examples: list[ArticleExample] = []

        for record in _read_jsonl(Path(articles_path)):
            article_id = str(record.get("id", "")).strip()
            if not article_id:
                raise ValueError("Every article record requires a non-empty 'id'")
            label = labels.get(article_id)
            if label is None and require_labels:
                continue

            examples.append(
                ArticleExample(
                    article_id=article_id,
                    title=str(record.get("title", "")),
                    text=str(record.get("text", record.get("content", ""))),
                    topic_ids=label.topic_ids if label else (),
                    entity_ids=_relation_ids(record.get("entities", [])),
                    linked_article_ids=_relation_ids(record.get("links", [])),
                    metadata={
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "content",
                            "entities",
                            "id",
                            "links",
                            "text",
                            "title",
                        }
                    },
                )
            )
        return cls(examples=examples, taxonomy=taxonomy)

    def _validate(self) -> None:
        article_ids: set[str] = set()
        valid_topics = set(self.taxonomy.topics)
        for example in self.examples:
            if example.article_id in article_ids:
                raise ValueError(f"Duplicate article ID: {example.article_id}")
            article_ids.add(example.article_id)
            unknown_topics = set(example.topic_ids) - valid_topics
            if unknown_topics:
                values = ", ".join(sorted(unknown_topics))
                raise ValueError(f"Unknown topic IDs for {example.article_id}: {values}")


def _load_labels(path: Path) -> dict[str, ArticleTopicLabel]:
    labels: dict[str, ArticleTopicLabel] = {}
    for record in _read_jsonl(path):
        article_id = str(record.get("article_id", "")).strip()
        if not article_id:
            raise ValueError("Every label record requires a non-empty 'article_id'")
        if article_id in labels:
            raise ValueError(f"Duplicate label record: {article_id}")
        labels[article_id] = ArticleTopicLabel(
            article_id=article_id,
            topic_ids=tuple(dict.fromkeys(str(value) for value in record.get("topic_ids", []))),
            confidence=float(record.get("confidence", 1.0)),
            provenance=tuple(record.get("provenance", [])),
        )
    return labels


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)
    return records


def _relation_ids(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    relation_ids: list[str] = []
    for value in values:
        if isinstance(value, dict):
            relation_id = value.get("id") or value.get("target_id") or value.get("title")
        else:
            relation_id = value
        if relation_id:
            relation_ids.append(str(relation_id))
    return tuple(dict.fromkeys(relation_ids))
