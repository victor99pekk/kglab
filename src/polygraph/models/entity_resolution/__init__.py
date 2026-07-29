"""Entity resolution trainer — binary classifier for merging duplicate entities.

Given a knowledge graph, generates positive/negative entity mention pairs and
trains a binary classifier to predict whether two mentions refer to the same
real-world entity. The trained model can then be used as a drop-in replacement
for string-based or embedding-based resolution in any pipeline.

Usage:
    from polygraph.models.entity_resolution import EntityResolutionTrainer

    trainer = EntityResolutionTrainer(
        kg_path="experiments/001_baseline/outputs/knowledge_graph.json",
        output_dir="experiments/training/001_er/outputs/",
        epochs=50,
    )
    trainer.run()
"""

from polygraph.models.entity_resolution.config import EntityResolutionConfig
from polygraph.models.entity_resolution.dataset import EntityPairDataset
from polygraph.models.entity_resolution.model import EntityResolutionModel
from polygraph.models.entity_resolution.train import EntityResolutionTrainer

__all__ = [
    "EntityPairDataset",
    "EntityResolutionConfig",
    "EntityResolutionModel",
    "EntityResolutionTrainer",
]
