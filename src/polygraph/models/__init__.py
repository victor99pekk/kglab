"""ML models for improving knowledge graph quality.

Trainable components that can be plugged into pipelines to improve
extraction, resolution, and graph structure.

Sub-packages:
    entity_resolution/   — binary classifier for merging duplicate entities
    link_prediction/     — score candidate triples for missing edges
    node_classification/ — predict entity types and attributes

Usage:
    from polygraph.models import BaseTrainer
    from polygraph.models.entity_resolution import EntityResolutionTrainer
"""

from polygraph.models._base import BaseTrainer
from polygraph.models.training_utils import (
    EarlyStopping,
    MetricTracker,
    SaveBest,
    count_parameters,
)

__all__ = [
    "BaseTrainer",
    "EarlyStopping",
    "MetricTracker",
    "SaveBest",
    "count_parameters",
]
