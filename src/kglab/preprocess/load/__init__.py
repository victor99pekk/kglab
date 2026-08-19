"""Data loading — baseline method.

Available methods:
    baseline  — File-based loader (txt, json, jsonl, csv)
    mongo     — MongoDB document archive loader
"""

from .baseline import DataLoader

__all__ = ["DataLoader"]
