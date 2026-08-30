"""Pipeline variants — each is a complete, swappable KG generation pipeline.

New pipeline variants should be registered in ``PIPELINE_REGISTRY`` so they
are discoverable by ``main.py`` and ``BenchmarkRunner`` via their string name.
"""

from kglab.pipelines._base import Pipeline
from kglab.pipelines.baseline import Baseline
from kglab.pipelines.local import Semantic
from kglab.pipelines.streaming import StreamingPipeline

#: Registry mapping variant name strings → Pipeline subclasses.
#: Add new entries here when creating a new pipeline variant::
#:
#:     from kglab.pipelines.llm.graphgen import GraphGen
#:     PIPELINE_REGISTRY["graphgen"] = GraphGen
PIPELINE_REGISTRY: dict[str, type[Pipeline]] = {
    # "baseline" is the canonical name used by ExperimentConfig defaults and
    # the example YAML configs; "surface" is a legacy alias kept for the
    # comparison tooling (compare_variants / compare_matrix).
    "baseline": Baseline,
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
