"""Model inference tools — thin wrappers that load trained checkpoints.

This module contains ONLY inference code. Training code (architectures,
datasets, training loops) lives in ``src/ml/`` — the parallel package.

Usage:
    from polygraph.models import ModelRegistry

    # Load a trained checkpoint
    tool = ModelRegistry.load(
        "entity_resolution",
        "experiments/ML_models/001_er/models/best.pt",
    )

    # Use it in a pipeline
    is_match = tool.resolve(entity_a, entity_b)
"""

# Import inference tools so they auto-register with ModelRegistry
import polygraph.models.entity_resolution  # noqa: F401 — side-effect: registers tool
import polygraph.models.node_classification  # noqa: F401 — side-effect: registers tool
from polygraph.models.registry import ModelRegistry

__all__ = ["ModelRegistry"]
