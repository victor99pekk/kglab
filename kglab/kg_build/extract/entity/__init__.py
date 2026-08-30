"""Entity extraction methods."""

from .en.spacy import SpacyExtractor
from .regex import SimpleExtractor

__all__ = ["SimpleExtractor", "SpacyExtractor"]
