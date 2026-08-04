"""Pipeline variants — each is a complete, swappable KG generation pipeline.

New pipeline variants should be registered in ``PIPELINE_REGISTRY`` so they
are discoverable by ``main.py`` and ``BenchmarkRunner`` via their string name.
"""

from polygraph.pipelines._base import Pipeline
from polygraph.pipelines.baseline import Baseline
from polygraph.pipelines.local import Semantic
from polygraph.pipelines.streaming import StreamingPipeline

#: Registry mapping variant name strings → Pipeline subclasses.
#: Add new entries here when creating a new pipeline variant::
#:
#:     from polygraph.pipelines.llm.graphgen import GraphGen
#:     PIPELINE_REGISTRY["graphgen"] = GraphGen
PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {
    "surface": Baseline,
    "semantic": Semantic,
}

__all__ = [
    "Baseline",
    "PIPELINE_REGISTRY",
    "Pipeline",
    "Semantic",
    "StreamingPipeline",
]
