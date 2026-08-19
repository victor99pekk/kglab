"""Node classification — GNN-based entity type prediction.

Given a knowledge graph, trains a GNN to predict the type of each entity
node (PERSON, ORG, GPE, ...) from its features and graph neighborhood.

The training code lives in ``src/ml/node_classification/``.
The inference-only tool that pipelines import lives in
``src/kglab/models/node_classification.py``.

Usage:
    from ml.node_classification import NodeClassificationTrainer

    trainer = NodeClassificationTrainer(
        kg_path="experiments/kg/001_baseline/outputs/knowledge_graph.json",
        output_dir="experiments/ML_models/001_nc/outputs/",
        epochs=100,
    )
    trainer.run()
"""

from ml.node_classification.config import NodeClassificationConfig
from ml.node_classification.dataset import NodeClassificationDataset
from ml.node_classification.models import MODEL_REGISTRY, available_variants, get_model
from ml.node_classification.train import NodeClassificationTrainer

__all__ = [
    "MODEL_REGISTRY",
    "NodeClassificationConfig",
    "NodeClassificationDataset",
    "NodeClassificationTrainer",
    "available_variants",
    "get_model",
]
