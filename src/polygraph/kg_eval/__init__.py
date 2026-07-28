"""KG Evaluation: measure quality, structure, coverage, and generate QA pairs.

Public API:
    from polygraph.kg_eval import QualityEvaluator, StructuralAuditor, ...
"""

from polygraph.kg_eval.metrics import QualityEvaluator
from polygraph.kg_eval.structural import StructuralAuditor

__all__ = [
    "QualityEvaluator",
    "StructuralAuditor",
]
