"""Text chunking methods.

Available methods:
    fixed     — Fixed-size character chunking with overlap
    sentence  — Sentence-aware, token-budgeted chunking
    semantic  — Semantic boundary detection via embeddings
"""

from .fixed import TextChunker
from .semantic import SemanticChunker
from .sentence import SentenceChunker

__all__ = ["TextChunker", "SentenceChunker", "SemanticChunker"]
