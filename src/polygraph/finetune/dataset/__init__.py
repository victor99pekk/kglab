"""Dataset generation for LLM fine-tuning.

Available methods:
    generator  — QA pair generation from KG + raw text
"""

from ._loaders import load_kg, load_raw_documents
from .generator import QADatasetGenerator

__all__ = ["QADatasetGenerator", "load_kg", "load_raw_documents"]
