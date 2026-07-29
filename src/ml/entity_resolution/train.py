"""Entity resolution trainer — full training loop.

Wires together the dataset, model, and training utilities to produce a
trained entity resolution classifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ml.base_trainer import BaseTrainer
from ml.entity_resolution.config import EntityResolutionConfig
from ml.entity_resolution.dataset import EntityPairDataset
from ml.entity_resolution.model import EntityResolutionModel
from ml.training_utils import (
    EarlyStopping,
    MetricTracker,
    SaveBest,
)

logger = logging.getLogger(__name__)


class EntityResolutionTrainer(BaseTrainer):
    """Train a binary classifier for entity resolution.

    Args:
        kg_path: Path to a knowledge graph JSON file.
        output_dir: Directory for outputs (models/, training_metrics.json, etc.).
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Optimizer learning rate.
        seed: Random seed.
        use_gpu: Whether to use GPU if available.
        model_config: EntityResolutionConfig with architecture & data settings.
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
        model_config: EntityResolutionConfig | None = None,
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
        self.model_config = model_config or EntityResolutionConfig()

    # ── BaseTrainer contract ───────────────────────────────────

    def prepare_data(self) -> None:
        """Load KG, generate entity pairs, and split into train/val/test."""
        dataset = EntityPairDataset(
            kg_path=self.kg_path,
            min_entity_frequency=self.model_config.min_entity_frequency,
            negative_ratio=self.model_config.negative_ratio,
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

        print(f"[data] {len(train)} train pairs, {len(val)} val pairs, {len(test)} test pairs")

    def build_model(self) -> None:
        """Instantiate the entity resolution model."""
        self.model = EntityResolutionModel(
            input_dim=self.model_config.embedding_dim,
            hidden_dims=self.model_config.hidden_dims,
            dropout=self.model_config.dropout,
        )
        print(f"[model] EntityResolutionModel(input_dim={self.model_config.embedding_dim})")

    def train(self) -> None:
        """Run the training loop with early stopping and metric tracking.

        This is a **skeleton** training loop. Replace the inner ``_train_epoch``
        and ``_validate`` calls with real PyTorch / sklearn logic.
        """
        early_stop = EarlyStopping(patience=10, mode="min")
        tracker = MetricTracker(self.output_dir, ["loss", "accuracy", "val_loss", "val_accuracy"])
        save_best = SaveBest(self.model_dir, filename="best.pt", mode="min")

        for epoch in range(1, self.epochs + 1):
            # ── Replace these stubs with real training logic ──
            train_loss = self._train_epoch(epoch)
            train_acc = 0.0  # placeholder
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

            if epoch % 5 == 0 or epoch == 1:
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
        """Evaluate on the test set. Returns metric dict.

        Override with real evaluation logic against ``self.test_data``.
        """
        # Placeholder — replace with real test-set evaluation
        return {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "train_pairs": len(self.train_data) if self.train_data else 0,
            "val_pairs": len(self.val_data) if self.val_data else 0,
            "test_pairs": len(self.test_data) if self.test_data else 0,
        }

    # ── Internal (stubs — replace with real logic) ─────────────

    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch. Returns average loss. (Stub.)"""
        _ = epoch
        return 0.0

    def _validate(self) -> tuple[float, float]:
        """Validate on val set. Returns (loss, accuracy). (Stub.)"""
        return 0.0, 0.0
