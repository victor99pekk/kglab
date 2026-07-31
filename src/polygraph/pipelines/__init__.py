"""Pipeline variants — each is a complete, swappable KG generation pipeline.

New pipeline variants should be registered in ``PIPELINE_REGISTRY`` so they
are discoverable by ``main.py`` and ``BenchmarkRunner`` via their string name.
"""

from polygraph.pipelines._base import Pipeline
from polygraph.pipelines._streaming import StreamingPipeline
from polygraph.pipelines.baseline import Baseline

#: Registry mapping variant name strings → Pipeline subclasses.
#: Add new entries here when creating a new pipeline variant::
#:
#:     from polygraph.pipelines.llm_extraction import LLMExtraction
#:     PIPELINE_REGISTRY["llm_extraction"] = LLMExtraction
PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {
    "baseline": Baseline,
}

__all__ = ["Baseline", "PIPELINE_REGISTRY", "Pipeline", "StreamingPipeline"]
