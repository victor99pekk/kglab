"""Leakage-safe heterogeneous graph construction for topic prediction."""

from __future__ import annotations

from dataclasses import dataclass

from ml.topic_classification.dataset import TopicClassificationDataset

EdgeType = tuple[str, str, str]
Edge = tuple[str, str]


@dataclass(frozen=True)
class IndexedTopicGraph:
    """Integer-indexed graph consumable by tensor frameworks."""

    node_ids: dict[str, tuple[str, ...]]
    node_index: dict[str, dict[str, int]]
    edge_indices: dict[EdgeType, tuple[tuple[int, ...], tuple[int, ...]]]
    target_indices: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HeterogeneousTopicGraph:
    """Typed graph with supervised edges stored outside message-passing edges."""

    node_ids: dict[str, tuple[str, ...]]
    edges: dict[EdgeType, tuple[Edge, ...]]
    targets: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if any(relation == "has_topic" for _, relation, _ in self.edges):
            raise ValueError("'has_topic' is supervised and cannot be a message-passing edge")

    def indexed(self) -> IndexedTopicGraph:
        """Convert stable string IDs into compact per-type integer indices."""
        node_index = {
            node_type: {node_id: index for index, node_id in enumerate(ids)}
            for node_type, ids in self.node_ids.items()
        }
        edge_indices: dict[EdgeType, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for edge_type, values in self.edges.items():
            source_type, _, target_type = edge_type
            source_indices = tuple(node_index[source_type][source] for source, _ in values)
            target_indices = tuple(node_index[target_type][target] for _, target in values)
            edge_indices[edge_type] = (source_indices, target_indices)

        targets = tuple(
            (node_index["article"][article_id], node_index["topic"][topic_id])
            for article_id, topic_id in self.targets
        )
        return IndexedTopicGraph(
            node_ids=self.node_ids,
            node_index=node_index,
            edge_indices=edge_indices,
            target_indices=targets,
        )


def build_topic_graph(
    dataset: TopicClassificationDataset,
    *,
    entity_relations: tuple[Edge, ...] = (),
) -> HeterogeneousTopicGraph:
    """Build initial Article/Entity/Topic/Domain graph.

    Article-topic labels become ``targets`` only. They never enter ``edges``,
    preventing direct label leakage during GNN message passing.
    """
    article_ids = tuple(example.article_id for example in dataset.examples)
    article_id_set = set(article_ids)
    entity_ids = tuple(
        sorted({entity_id for example in dataset.examples for entity_id in example.entity_ids})
    )
    topic_ids = tuple(dataset.taxonomy.topics)
    domain_ids = tuple(dataset.taxonomy.domains)

    mentions = tuple(
        (example.article_id, entity_id)
        for example in dataset.examples
        for entity_id in example.entity_ids
    )
    links = tuple(
        (example.article_id, target_id)
        for example in dataset.examples
        for target_id in example.linked_article_ids
        if target_id in article_id_set and target_id != example.article_id
    )
    topic_domains = tuple(
        (topic.topic_id, topic.domain_id) for topic in dataset.taxonomy.topics.values()
    )
    targets = tuple(
        (example.article_id, topic_id)
        for example in dataset.examples
        for topic_id in example.topic_ids
    )

    edges: dict[EdgeType, tuple[Edge, ...]] = {
        ("article", "mentions", "entity"): mentions,
        ("entity", "mentioned_by", "article"): _reverse(mentions),
        ("article", "links_to", "article"): links,
        ("article", "linked_from", "article"): _reverse(links),
        ("topic", "in_domain", "domain"): topic_domains,
        ("domain", "contains_topic", "topic"): _reverse(topic_domains),
    }
    if entity_relations:
        entity_id_set = set(entity_ids)
        invalid = [
            edge
            for edge in entity_relations
            if edge[0] not in entity_id_set or edge[1] not in entity_id_set
        ]
        if invalid:
            raise ValueError(f"Entity relations reference unknown nodes: {invalid[:3]}")
        edges[("entity", "related_to", "entity")] = entity_relations

    return HeterogeneousTopicGraph(
        node_ids={
            "article": article_ids,
            "entity": entity_ids,
            "topic": topic_ids,
            "domain": domain_ids,
        },
        edges=edges,
        targets=targets,
    )


def _reverse(edges: tuple[Edge, ...]) -> tuple[Edge, ...]:
    return tuple((target, source) for source, target in edges)
