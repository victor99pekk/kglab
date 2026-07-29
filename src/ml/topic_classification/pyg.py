"""PyTorch Geometric adapter for the framework-neutral topic graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from ml.topic_classification.graph import HeterogeneousTopicGraph


def to_hetero_data(
    graph: HeterogeneousTopicGraph,
    features: Mapping[str, Tensor],
) -> HeteroData:
    """Convert stable graph IDs and injected node features into ``HeteroData``.

    Article-topic supervision becomes ``data["article"].y``. No
    ``("article", "has_topic", "topic")`` edge store is created, so the
    labels cannot accidentally enter message passing.
    """
    indexed = graph.indexed()
    data = HeteroData()

    for node_type, node_ids in indexed.node_ids.items():
        if node_type not in features:
            raise ValueError(f"Missing features for node type: {node_type}")
        node_features = features[node_type]
        if node_features.ndim != 2:
            raise ValueError(f"{node_type} features must be a rank-2 tensor")
        if node_features.shape[0] != len(node_ids):
            raise ValueError(
                f"{node_type} has {len(node_ids)} nodes but "
                f"{node_features.shape[0]} feature rows"
            )
        data[node_type].x = node_features
        data[node_type].node_ids = list(node_ids)

    for edge_type, (source_indices, target_indices) in indexed.edge_indices.items():
        data[edge_type].edge_index = torch.tensor(
            [source_indices, target_indices],
            dtype=torch.long,
        )

    article_count = len(indexed.node_ids["article"])
    topic_count = len(indexed.node_ids["topic"])
    labels = torch.zeros((article_count, topic_count), dtype=torch.float32)
    for article_index, topic_index in indexed.target_indices:
        labels[article_index, topic_index] = 1.0
    data["article"].y = labels

    if any(relation == "has_topic" for _, relation, _ in data.edge_types):
        raise RuntimeError("'has_topic' leaked into PyG message-passing edges")
    return data


__all__ = ["to_hetero_data"]
