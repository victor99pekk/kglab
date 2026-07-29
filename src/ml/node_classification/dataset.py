"""Node classification dataset — loads entities from a KG and assigns labels.

Labels are derived from the entity ``type`` field (PERSON, ORG, GPE, ...)
as defined in the ontology. The dataset produces entity → integer label
mappings for training a GNN classifier.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


class NodeClassificationDataset:
    """Labeled entity nodes for training a node classifier.

    Args:
        kg_path: Path to a knowledge graph JSON file.
        min_entity_frequency: Entity types with fewer examples are dropped.
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        kg_path: str | Path,
        min_entity_frequency: int = 2,
        seed: int = 42,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.min_entity_frequency = min_entity_frequency
        self.seed = seed

        # Populated by load()
        self.nodes: list[dict[str, Any]] = []
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        self.num_classes: int = 0

    # ── Public API ──────────────────────────────────────────────

    def load(self) -> NodeClassificationDataset:
        """Load the KG, extract entity nodes, and build label mappings.

        Returns self for method chaining.
        """
        entities = self._load_entities()
        self.nodes = self._filter_and_label(entities)
        return self

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split nodes into train / validation / test sets.

        Splits are stratified — each label class is split independently
        so the train/val/test distribution matches the full dataset.

        Returns:
            (train_nodes, val_nodes, test_nodes) — each node has ``label_idx``
            set to the integer class label.
        """
        rng = random.Random(self.seed)

        # Group nodes by label for stratified split
        by_label: dict[int, list[dict[str, Any]]] = {i: [] for i in range(self.num_classes)}
        for node in self.nodes:
            by_label[node["label_idx"]].append(node)

        train: list[dict[str, Any]] = []
        val: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []

        for _label_idx, group in by_label.items():
            rng.shuffle(group)
            n = len(group)
            train_end = max(1, int(n * train_ratio))
            val_end = train_end + max(1, int(n * val_ratio))

            train.extend(group[:train_end])
            val.extend(group[train_end:val_end])
            test.extend(group[val_end:])

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        return train, val, test

    # ── Internal ────────────────────────────────────────────────

    def _load_entities(self) -> list[dict[str, Any]]:
        """Load entity nodes from the KG JSON file."""
        with open(self.kg_path, encoding="utf-8") as f:
            kg = json.load(f)

        if "entities" in kg:
            return kg["entities"]
        if "graph" in kg:
            nodes = kg["graph"].get("nodes", [])
            return [n for n in nodes if n.get("type") not in ("Chunk", "Document")]
        return []

    def _filter_and_label(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter entities by type frequency and assign integer labels.

        Entity types (PERSON, ORG, GPE, ...) are sorted alphabetically and
        mapped to 0..K-1. Types with fewer than ``min_entity_frequency``
        examples are excluded.
        """
        # Count type occurrences
        type_counts: Counter[str] = Counter()
        for ent in entities:
            ent_type = ent.get("type", "").strip()
            if ent_type:
                type_counts[ent_type] += 1

        # Build label mapping from types that meet the frequency threshold
        valid_types = sorted(t for t, c in type_counts.items() if c >= self.min_entity_frequency)
        self.label_to_idx = {t: i for i, t in enumerate(valid_types)}
        self.idx_to_label = {i: t for t, i in self.label_to_idx.items()}
        self.num_classes = len(valid_types)

        # Filter and label entities
        labeled: list[dict[str, Any]] = []
        for ent in entities:
            ent_type = ent.get("type", "").strip()
            if ent_type in self.label_to_idx:
                ent_copy = dict(ent)
                ent_copy["label_idx"] = self.label_to_idx[ent_type]
                ent_copy["label_name"] = ent_type
                labeled.append(ent_copy)

        return labeled
