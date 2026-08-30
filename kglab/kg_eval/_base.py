"""Abstract base class for all KG evaluators.

All evaluators must implement:
    evaluate(graph, entities, triples) -> dict[str, Any]

This contract enables the pipeline to dispatch to any evaluator generically
and lets users write custom evaluators that plug in without modifying core code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx


class BaseEvaluator(ABC):
    """Abstract base for all KG evaluators.

    Subclasses must implement ``evaluate()``. Constructor signatures are
    free — inject whatever dependencies the evaluator needs (LLM client,
    ontology path, thresholds, etc.) via ``__init__``.

    Usage:
        class MyEvaluator(BaseEvaluator):
            def __init__(self, threshold: float = 0.5) -> None:
                self.threshold = threshold

            def evaluate(self, graph, entities, triples) -> dict:
                return {"my_metric": compute_something(...)}
    """

    @abstractmethod
    def evaluate(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Run all metrics and return a flat report dict.

        Args:
            graph: A ``networkx.DiGraph`` of the knowledge graph.
            entities: List of entity dicts (each has ``name``, ``type``, etc.).
            triples: List of ``(subject, predicate, object, source_text)`` tuples.

        Returns:
            A flat dictionary of metric names to scores. The exact keys
            depend on the evaluator, but every implementation should include
            enough context for a human to interpret the result.
        """
        ...
