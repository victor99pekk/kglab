"""Abstract base trainer — subclass for each ML task.

Mirrors the ``Pipeline`` base class pattern: define a contract (prepare_data,
build_model, train, evaluate) and let subclasses fill in the details.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTrainer(ABC):
    """Abstract trainer with a standard lifecycle.

    Subclass and override each abstract method to create a trainer for a
    specific ML task (entity resolution, link prediction, etc.).

    Usage::

        class MyTrainer(BaseTrainer):
            def prepare_data(self): ...
            def build_model(self): ...
            def train(self): ...
            def evaluate(self): ...

        trainer = MyTrainer(
            kg_path="experiments/kg/001_baseline/outputs/knowledge_graph.json",
            output_dir="experiments/ML_models/001_er/outputs/",
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
        )
        trainer.run()
    """

    def __init__(
        self,
        kg_path: str | Path,
        output_dir: str | Path,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        seed: int = 42,
        use_gpu: bool = True,
        **kwargs: Any,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / "models"
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.use_gpu = use_gpu
        self._extra = kwargs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Subclasses populate these during prepare_data / build_model
        self.model: Any = None
        self.train_data: Any = None
        self.val_data: Any = None
        self.test_data: Any = None

        # Training history
        self.history: dict[str, list[float]] = {}

    # ── Abstract contract ──────────────────────────────────────

    @abstractmethod
    def prepare_data(self) -> None:
        """Load the KG, generate training samples, split train/val/test."""
        ...

    @abstractmethod
    def build_model(self) -> None:
        """Instantiate the model architecture."""
        ...

    @abstractmethod
    def train(self) -> None:
        """Run the training loop and save checkpoints."""
        ...

    @abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the test set and return metric dict."""
        ...

    # ── Orchestration ──────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Full training lifecycle: prepare → build → train → evaluate.

        Returns a result dict suitable for writing to ``training_metrics.json``.
        """
        print(f"=== {self.__class__.__name__} ===")
        print(f"KG:     {self.kg_path}")
        print(f"Output: {self.output_dir}\n")

        t0 = time.perf_counter()

        self.prepare_data()
        self.build_model()
        self.train()
        metrics = self.evaluate()

        elapsed_s = round(time.perf_counter() - t0, 2)
        metrics["wall_time_s"] = elapsed_s

        # Write training metrics
        metrics_path = self.output_dir / "training_metrics.json"
        payload = {
            "task": self.__class__.__name__,
            "kg_path": str(self.kg_path),
            "hyperparameters": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
            "metrics": metrics,
            "history": self.history,
        }
        metrics_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n[metrics] → {metrics_path}")
        return payload
