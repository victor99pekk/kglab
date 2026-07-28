"""KG Evaluation: measure quality, structure, coverage.

Function API:
    from polygraph.kg_eval import metrics, structural

    report = metrics.evaluate(graph, entities, triples)
    audit = structural.run(graph, entities, triples, ontology_path="configs/default_ontology.yaml")

Class API:
    from polygraph.kg_eval import QualityEvaluator, StructuralAuditor
"""

from types import SimpleNamespace

from polygraph.kg_eval.metrics import QualityEvaluator
from polygraph.kg_eval.structural import StructuralAuditor

# ── Function API ────────────────────────────────────────────────

metrics = SimpleNamespace()
metrics.evaluate = lambda graph, entities, triples: (
    QualityEvaluator().evaluate_graph(graph, entities, triples)
)
metrics.completeness = lambda entities: QualityEvaluator().completeness(entities)

structural = SimpleNamespace()
structural.run = lambda graph, entities, triples, ontology_path=None: (
    StructuralAuditor(ontology_path=ontology_path).audit(graph, entities, triples)
)

__all__ = [
    "QualityEvaluator",
    "StructuralAuditor",
    "metrics",
    "structural",
]
