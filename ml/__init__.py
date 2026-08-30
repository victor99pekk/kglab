"""ML training library — model architectures, training loops, and utilities.

This package lives **parallel** to ``kglab``. It contains all training-
time code: model definitions, dataset preparation, training loops, and
shared utilities. Pipelines should NOT import from here.

Once a model is trained, the inference-only tool in ``kglab.models``
loads the checkpoint and exposes a clean public API.

Sub-packages:
    entity_resolution/   — train a binary classifier for merging duplicate entities
    link_prediction/     — train a model to score candidate triples (future)
    node_classification/ — train a model to predict entity types (future)

Usage:
    from ml.base_trainer import BaseTrainer
    from ml.entity_resolution import EntityResolutionTrainer
"""

from ml.base_trainer import BaseTrainer
from ml.training_utils import (
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
