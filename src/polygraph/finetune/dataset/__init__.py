"""Dataset generation for LLM fine-tuning.

Available methods:
    generator  — QA pair generation from KG + raw text
"""

from .generator import QADatasetGenerator

__all__ = ["QADatasetGenerator"]
