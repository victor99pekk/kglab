"""Node classification trainer — full training loop for GNN entity type prediction.

Wires together the dataset, model registry, and training utilities to
train a GNN that predicts entity types (PERSON, ORG, GPE, ...) from
entity features and graph structure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ml.base_trainer import BaseTrainer
from ml.node_classification.config import NodeClassificationConfig
from ml.node_classification.dataset import NodeClassificationDataset
from ml.node_classification.models import get_model
from ml.training_utils import (
    EarlyStopping,
    MetricTracker,
    SaveBest,
)

logger = logging.getLogger(__name__)


class NodeClassificationTrainer(BaseTrainer):
    """Train a GNN to classify entity nodes by type.

    Args:
        kg_path: Path to a knowledge graph JSON file.
        output_dir: Directory for outputs (models/, training_metrics.json, etc.).
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed.
        use_gpu: Whether to use GPU if available.
        model_config: NodeClassificationConfig with architecture settings.
    """

    def __init__(
        self,
        kg_path: str | Path,
        output_dir: str | Path,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        seed: int = 42,
        use_gpu: bool = True,
        model_config: NodeClassificationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kg_path=kg_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            use_gpu=use_gpu,
            **kwargs,
        )
        self.model_config = model_config or NodeClassificationConfig()

    # ── BaseTrainer contract ───────────────────────────────────

    def prepare_data(self) -> None:
        """Load KG, extract labeled nodes, and split into train/val/test."""
        dataset = NodeClassificationDataset(
            kg_path=self.kg_path,
            min_entity_frequency=self.model_config.min_entity_frequency,
            seed=self.seed,
        )
        dataset.load()

        train, val, test = dataset.split(
            train_ratio=self.model_config.train_split,
            val_ratio=self.model_config.val_split,
        )

        self.train_data = train
        self.val_data = val
        self.test_data = test

        # Store label metadata for the model
        self._num_classes = dataset.num_classes
        self._label_names = dataset.idx_to_label

        print(
            f"[data] {len(train)} train / {len(val)} val / {len(test)} test nodes, "
            f"{self._num_classes} classes: {list(self._label_names.values())}"
        )

    def build_model(self) -> None:
        """Instantiate the GNN via the model registry."""
        model_cls = get_model(self.model_config.model_variant)

        from dataclasses import asdict

        model_kwargs = asdict(self.model_config)
        model_kwargs.pop("model_variant", None)
        model_kwargs["num_classes"] = self._num_classes

        self.model = model_cls(**model_kwargs)
        print(
            f"[model] {model_cls.__name__} "
            f"(variant={self.model_config.model_variant}, "
            f"num_classes={self._num_classes}, "
            f"hidden_dim={self.model_config.hidden_dim})"
        )

    def train(self) -> None:
        """Run the training loop with early stopping and metric tracking.

        This is a **skeleton** — replace _train_epoch and _validate with
        real PyTorch / PyTorch Geometric logic.
        """
        early_stop = EarlyStopping(patience=15, mode="min")
        tracker = MetricTracker(self.output_dir, ["loss", "accuracy", "val_loss", "val_accuracy"])
        save_best = SaveBest(self.model_dir, filename="best.pt", mode="min")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            train_acc = 0.0
            val_loss, val_acc = self._validate()

            tracker.log(
                epoch,
                loss=train_loss,
                accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
            self.history.setdefault("loss", []).append(train_loss)
            self.history.setdefault("val_loss", []).append(val_loss)

            improved = early_stop(val_loss)
            if improved:
                save_best(self.model, val_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[epoch {epoch:3d}/{self.epochs}] "
                    f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_acc={val_acc:.4f}"
                )

            if early_stop.should_stop:
                print(f"[early_stop] No improvement for {early_stop.patience} epochs. Stopping.")
                break

        tracker.save()
        print(f"[train] best_val_loss={early_stop.best_score:.4f}")

    def evaluate(self) -> dict[str, float]:
        """Evaluate on the test set. Returns metric dict."""
        return {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "test_macro_f1": 0.0,
            "num_classes": self._num_classes,
            "train_nodes": len(self.train_data) if self.train_data else 0,
            "val_nodes": len(self.val_data) if self.val_data else 0,
            "test_nodes": len(self.test_data) if self.test_data else 0,
        }

    # ── Internal stubs (replace with real PyTorch logic) ───────

    def _train_epoch(self, epoch: int) -> float:
        _ = epoch
        return 0.0

    def _validate(self) -> tuple[float, float]:
        return 0.0, 0.0
