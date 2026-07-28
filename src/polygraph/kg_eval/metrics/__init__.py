"""KG quality metrics.

Available methods:
    quality  — Graph quality scoring (completeness, consistency, duplication, etc.)
"""

from .quality import QualityEvaluator

__all__ = ["QualityEvaluator"]
