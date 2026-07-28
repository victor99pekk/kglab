"""Pipeline variants — each is a complete, swappable KG generation pipeline."""

from polygraph.pipelines._base import Pipeline
from polygraph.pipelines.baseline import Baseline

__all__ = ["Baseline", "Pipeline"]
