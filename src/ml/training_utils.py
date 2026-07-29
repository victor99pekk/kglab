"""Shared training utilities — early stopping, metric tracking, checkpointing.

These are framework-agnostic helpers that work with any PyTorch / sklearn /
custom model. Import them into individual task trainers to avoid duplicating
boilerplate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``"min"`` (lower is better, e.g. loss) or ``"max"`` (higher is
              better, e.g. accuracy).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float = math.inf if mode == "min" else -math.inf
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, current: float) -> bool:
        if self.mode == "min":
            improved = current < self.best_score - self.min_delta
        else:
            improved = current > self.best_score + self.min_delta

        if improved:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return improved


class MetricTracker:
    """Accumulate per-epoch metrics and write a running log.

    Args:
        output_dir: Directory to write ``training_log.json`` into.
        metrics: List of metric names to track (e.g. ``["loss", "accuracy"]``).
    """

    def __init__(self, output_dir: str | Path, metrics: list[str]) -> None:
        self.output_dir = Path(output_dir)
        self.metrics = metrics
        self.records: list[dict[str, float]] = []

    def log(self, epoch: int, **values: float) -> None:
        """Record metrics for a single epoch."""
        entry = {"epoch": epoch}
        for name in self.metrics:
            entry[name] = values.get(name, float("nan"))
        self.records.append(entry)

    def save(self) -> Path:
        """Persist all logged records to disk."""
        path = self.output_dir / "training_log.json"
        path.write_text(json.dumps(self.records, indent=2))
        return path

    def latest(self) -> dict[str, float]:
        """Return the most recent record (empty dict if none)."""
        return self.records[-1] if self.records else {}


class SaveBest:
    """Save model checkpoint only when a monitored metric improves.

    This is a **protocol** — it calls ``model.save(path)``. Your model must
    implement a ``save(path: str | Path)`` method. Works with PyTorch (save
    via ``torch.save``), sklearn (joblib), or any custom model.

    Args:
        model_dir: Directory to save checkpoints into.
        filename: Name for the best checkpoint file (e.g. ``"best.pt"``).
        mode: ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        model_dir: str | Path,
        filename: str = "best.pt",
        mode: str = "min",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.filename = filename
        self.mode = mode
        self.best_score: float = math.inf if mode == "min" else -math.inf

    def __call__(self, model: Any, current: float) -> bool:
        """Save model if *current* is the best seen so far. Returns True if saved."""
        is_best = current < self.best_score if self.mode == "min" else current > self.best_score

        if is_best:
            self.best_score = current
            path = self.model_dir / self.filename
            if hasattr(model, "save"):
                model.save(str(path))
            return True
        return False


def count_parameters(model: Any) -> int:
    """Return the total number of trainable parameters (PyTorch-compatible).

    Falls back to ``-1`` for non-PyTorch models.
    """
    try:
        import torch
    except ImportError:
        return -1
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return -1
