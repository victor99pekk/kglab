"""Full-batch PyG training loop for multilabel article-topic prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch_geometric.data import HeteroData

from ml.base_trainer import BaseTrainer
from ml.topic_classification.config import TopicClassificationConfig
from ml.topic_classification.models import get_model
from ml.training_utils import EarlyStopping, MetricTracker, SaveBest


class TopicClassificationTrainer(BaseTrainer):
    """Train on a prepared ``HeteroData`` graph containing BERT node features.

    ``kg_path`` points to a serialized ``HeteroData`` artifact. Its article
    store must contain a dense multilabel matrix named ``y``.
    """

    def __init__(
        self,
        kg_path: str | Path,
        output_dir: str | Path,
        epochs: int = 100,
        learning_rate: float = 0.001,
        seed: int = 42,
        use_gpu: bool = True,
        model_config: TopicClassificationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kg_path=kg_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=0,
            learning_rate=learning_rate,
            seed=seed,
            use_gpu=use_gpu,
            **kwargs,
        )
        self.model_config = model_config or TopicClassificationConfig()
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.data: HeteroData | None = None
        self.optimizer: Adam | None = None
        self.criterion = nn.BCEWithLogitsLoss()

    def prepare_data(self) -> None:
        """Load prepared graph and create deterministic article masks."""
        data = torch.load(self.kg_path, map_location="cpu", weights_only=False)
        if not isinstance(data, HeteroData):
            raise TypeError("Topic trainer requires a serialized PyG HeteroData graph")
        if "y" not in data["article"]:
            raise ValueError("HeteroData article store requires multilabel target 'y'")

        article_count = data["article"].num_nodes
        if article_count < 3:
            raise ValueError("At least three labeled articles are required for train/val/test")
        if self.model_config.train_split + self.model_config.val_split >= 1:
            raise ValueError("train_split + val_split must be less than 1")

        generator = torch.Generator().manual_seed(self.seed)
        permutation = torch.randperm(article_count, generator=generator)
        train_end = max(1, int(article_count * self.model_config.train_split))
        val_end = min(
            article_count - 1,
            train_end + max(1, int(article_count * self.model_config.val_split)),
        )
        data["article"].train_mask = _mask(article_count, permutation[:train_end])
        data["article"].val_mask = _mask(article_count, permutation[train_end:val_end])
        data["article"].test_mask = _mask(article_count, permutation[val_end:])

        self.data = data.to(self.device)
        self.train_data = self.data["article"].train_mask
        self.val_data = self.data["article"].val_mask
        self.test_data = self.data["article"].test_mask

    def build_model(self) -> None:
        """Build selected model from PyG metadata and node feature sizes."""
        data = self._require_data()
        input_dims = {
            node_type: int(features.shape[1])
            for node_type, features in data.x_dict.items()
        }
        model_cls = get_model(self.model_config.model_variant)
        self.model = model_cls(
            input_dims=input_dims,
            edge_types=list(data.edge_types),
            hidden_dim=self.model_config.hidden_dim,
            num_layers=self.model_config.num_layers,
            dropout=self.model_config.dropout,
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self) -> None:
        """Optimize weighted multilabel logits with validation early stopping."""
        data = self._require_data()
        optimizer = self._require_optimizer()
        early_stop = EarlyStopping(patience=15, mode="min")
        tracker = MetricTracker(
            self.output_dir,
            ["loss", "micro_f1", "val_loss", "val_micro_f1"],
        )
        save_best = SaveBest(self.model_dir, filename="best.pt", mode="min")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(data.x_dict, data.edge_index_dict)
            train_mask = data["article"].train_mask
            train_loss = self.criterion(logits[train_mask], data["article"].y[train_mask])
            train_loss.backward()
            optimizer.step()

            train_metrics = _multilabel_metrics(
                logits[train_mask].detach(),
                data["article"].y[train_mask],
                self.model_config.threshold,
            )
            val_loss, val_metrics = self._evaluate_mask(data["article"].val_mask)
            tracker.log(
                epoch,
                loss=float(train_loss.item()),
                micro_f1=train_metrics["micro_f1"],
                val_loss=val_loss,
                val_micro_f1=val_metrics["micro_f1"],
            )
            self.history.setdefault("loss", []).append(float(train_loss.item()))
            self.history.setdefault("val_loss", []).append(val_loss)

            if early_stop(val_loss):
                save_best(self.model, val_loss)
            if early_stop.should_stop:
                break
        tracker.save()

    def evaluate(self) -> dict[str, float]:
        """Evaluate best in-memory model on held-out articles."""
        test_loss, metrics = self._evaluate_mask(self._require_data()["article"].test_mask)
        return {"test_loss": test_loss, **metrics}

    def _evaluate_mask(self, mask: Tensor) -> tuple[float, dict[str, float]]:
        data = self._require_data()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data.x_dict, data.edge_index_dict)
            loss = self.criterion(logits[mask], data["article"].y[mask])
            metrics = _multilabel_metrics(
                logits[mask],
                data["article"].y[mask],
                self.model_config.threshold,
            )
        return float(loss.item()), metrics

    def _require_data(self) -> HeteroData:
        if self.data is None:
            raise RuntimeError("Call prepare_data() before accessing graph data")
        return self.data

    def _require_optimizer(self) -> Adam:
        if self.optimizer is None:
            raise RuntimeError("Call build_model() before training")
        return self.optimizer


def _mask(size: int, indices: Tensor) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask


def _multilabel_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float,
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    expected = targets.bool()
    true_positive = (predictions & expected).sum().float()
    false_positive = (predictions & ~expected).sum().float()
    false_negative = (~predictions & expected).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    micro_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "micro_precision": float(precision.item()),
        "micro_recall": float(recall.item()),
        "micro_f1": float(micro_f1.item()),
    }
