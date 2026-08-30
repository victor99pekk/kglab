"""Entity resolution trainer — binary classifier for merging duplicate entities.

Given a knowledge graph, generates positive/negative entity mention pairs and
trains a binary classifier to predict whether two mentions refer to the same
real-world entity. The trained model can then be used as a drop-in replacement
for string-based or embedding-based resolution in any pipeline.

The training code lives here in ``ml/``. The inference-only tool that
pipelines import lives in ``kglab/models/entity_resolution.py``.

Usage:
    from ml.entity_resolution import EntityResolutionTrainer

    trainer = EntityResolutionTrainer(
        kg_path="experiments/kg/001_baseline/outputs/knowledge_graph.json",
        output_dir="experiments/ML_models/001_er/outputs/",
        epochs=50,
    )
    trainer.run()
"""

from ml.entity_resolution.config import EntityResolutionConfig
from ml.entity_resolution.dataset import EntityPairDataset
from ml.entity_resolution.models import MODEL_REGISTRY, available_variants, get_model
from ml.entity_resolution.train import EntityResolutionTrainer

__all__ = [
    "EntityPairDataset",
    "EntityResolutionConfig",
    "MODEL_REGISTRY",
    "available_variants",
    "get_model",
    "EntityResolutionTrainer",
]
