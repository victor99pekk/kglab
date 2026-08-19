"""Text normalization and cleaning — baseline method.

Available methods:
    baseline  — English text normalization (strip, quotes, dashes, unicode)
"""

from .baseline import TextCleaner, TextCleanerBackend
from .en.normalizer import EnglishCleaner

__all__ = ["EnglishCleaner", "TextCleaner", "TextCleanerBackend"]
