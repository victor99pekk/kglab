"""KG Evaluation: measure quality, structure, coverage.

Function API:
    from polygraph.kg_eval import metrics, structural

    report = metrics.evaluate(graph, entities, triples)
    audit = structural.run(graph, entities, triples, ontology_path="configs/default_ontology.yaml")

Class API:
    from polygraph.kg_eval import BaseEvaluator, QualityEvaluator, StructuralAuditor, AccuracyEvaluator
"""

from types import SimpleNamespace

from polygraph.kg_eval._base import BaseEvaluator
from polygraph.kg_eval.metrics import AccuracyEvaluator, QualityEvaluator
from polygraph.kg_eval.structural import StructuralAuditor

# ── Function API ────────────────────────────────────────────────

metrics = SimpleNamespace()
metrics.evaluate = lambda graph, entities, triples: (
    QualityEvaluator().evaluate(graph, entities, triples)
)
metrics.completeness = lambda entities: QualityEvaluator().completeness(entities)

structural = SimpleNamespace()
structural.run = lambda graph, entities, triples, ontology_path=None: (
    StructuralAuditor(ontology_path=ontology_path).evaluate(graph, entities, triples)
)

__all__ = [
    "AccuracyEvaluator",
    "BaseEvaluator",
    "QualityEvaluator",
    "StructuralAuditor",
    "metrics",
    "structural",
]
