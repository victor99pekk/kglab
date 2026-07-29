"""Training components for Level-1 Wikipedia topic classification.

The task is modeled as article-to-topic link prediction over a heterogeneous
graph.  This package owns training-time data and model code.  Production
inference wrappers belong in ``polygraph.models`` only after a checkpoint
format has been validated by experiments.

PyTorch Geometric conversion and training follow the task-package conventions
used by ``ml.node_classification``.
"""

from ml.topic_classification.config import TopicClassificationConfig
from ml.topic_classification.dataset import (
    ArticleExample,
    ArticleTopicLabel,
    TopicClassificationDataset,
    TopicDefinition,
    TopicTaxonomy,
)
from ml.topic_classification.graph import (
    EdgeType,
    HeterogeneousTopicGraph,
    IndexedTopicGraph,
    build_topic_graph,
)
from ml.topic_classification.pyg import to_hetero_data
from ml.topic_classification.train import TopicClassificationTrainer

__all__ = [
    "ArticleExample",
    "ArticleTopicLabel",
    "EdgeType",
    "HeterogeneousTopicGraph",
    "IndexedTopicGraph",
    "TopicClassificationConfig",
    "TopicClassificationDataset",
    "TopicClassificationTrainer",
    "TopicDefinition",
    "TopicTaxonomy",
    "build_topic_graph",
    "to_hetero_data",
]
