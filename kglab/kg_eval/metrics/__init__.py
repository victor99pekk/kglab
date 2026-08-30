"""KG quality metrics.

Available methods:
    quality   — Graph quality scoring (completeness, consistency, duplication, etc.)
    accuracy  — LLM-judge accuracy evaluation (semantic accuracy, triple classification)
"""

from .accuracy import AccuracyEvaluator
from .quality import QualityEvaluator

__all__ = ["AccuracyEvaluator", "QualityEvaluator"]
