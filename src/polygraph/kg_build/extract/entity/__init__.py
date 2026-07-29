"""Entity extraction methods."""

from .regex import SimpleExtractor
from .spacy import EnglishExtractor

__all__ = ["EnglishExtractor", "SimpleExtractor"]
