"""Deduplication methods.

Available methods:
    minhash   — MinHash LSH clustering (scalable, global)
    semantic  — Embedding-based semantic dedup (small datasets)
    layered   — Combines multiple methods (minhash → semantic)
"""

from .layered import Deduplicator
from .minhash import DuplicateAssignment, DuplicateMatch, GlobalDeduplicator
from .semantic import SemanticDeduplicator

__all__ = [
    "Deduplicator",
    "DuplicateAssignment",
    "DuplicateMatch",
    "GlobalDeduplicator",
    "SemanticDeduplicator",
]
