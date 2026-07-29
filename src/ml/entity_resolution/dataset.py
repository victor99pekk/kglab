"""Entity pair dataset — generates positive/negative mention pairs from a KG.

Strategy:
    1. Group entity nodes by their canonical name (case-folded + stripped).
    2. Positive pairs: any two mentions of the same canonical entity.
    3. Negative pairs: randomly sampled mentions of different canonical entities,
       with a configurable negative_ratio.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


class EntityPairDataset:
    """Positive/negative entity mention pairs for training a resolution classifier.

    Args:
        kg_path: Path to a knowledge graph JSON file (as exported by the pipeline).
        min_entity_frequency: Discard entities with fewer than this many mentions.
        negative_ratio: How many negative pairs to generate per positive pair.
        seed: Random seed for reproducible negative sampling.
    """

    def __init__(
        self,
        kg_path: str | Path,
        min_entity_frequency: int = 2,
        negative_ratio: int = 3,
        seed: int = 42,
    ) -> None:
        self.kg_path = Path(kg_path)
        self.min_entity_frequency = min_entity_frequency
        self.negative_ratio = negative_ratio
        self.seed = seed

        # Populated by load()
        self.pairs: list[dict[str, Any]] = []
        self.entity_groups: dict[str, list[dict[str, Any]]] = {}

    # ── Public API ──────────────────────────────────────────────

    def load(self) -> EntityPairDataset:
        """Load the KG, group entities, and generate train/val/test splits.

        Returns self for method chaining.
        """
        entities = self._load_entities()
        self.entity_groups = self._group_entities(entities)
        self.pairs = self._generate_pairs()
        return self

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Split pairs into train / validation / test sets.

        Returns:
            (train_pairs, val_pairs, test_pairs) — each is a list of dicts
            with keys ``entity_a``, ``entity_b``, ``label`` (1 = match, 0 = no match).
        """
        rng = random.Random(self.seed)
        shuffled = list(self.pairs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    # ── Internal ────────────────────────────────────────────────

    def _load_entities(self) -> list[dict[str, Any]]:
        """Load entity nodes from the KG JSON file."""
        with open(self.kg_path, encoding="utf-8") as f:
            kg = json.load(f)

        # Support both node-link and flat entity list formats
        if "entities" in kg:
            return kg["entities"]
        if "graph" in kg:
            nodes = kg["graph"].get("nodes", [])
            return [n for n in nodes if n.get("type") not in ("Chunk", "Document")]
        return []

    @staticmethod
    def _group_entities(
        entities: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group entities by case-folded, stripped name."""
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ent in entities:
            key = ent.get("name", "").strip().casefold()
            if key:
                groups[key].append(ent)
        return dict(groups)

    def _generate_pairs(self) -> list[dict[str, Any]]:
        """Generate positive (same group) and negative (different groups) pairs."""
        rng = random.Random(self.seed)
        groups = {
            k: v for k, v in self.entity_groups.items() if len(v) >= self.min_entity_frequency
        }
        if len(groups) < 2:
            return []

        group_keys = list(groups.keys())
        pairs: list[dict[str, Any]] = []

        # Positive pairs — mentions of the same canonical entity
        for _key, mentions in groups.items():
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    pairs.append(
                        {
                            "entity_a": mentions[i],
                            "entity_b": mentions[j],
                            "label": 1,
                        }
                    )

        # Negative pairs — mentions of different entities
        if pairs:
            for _ in range(len(pairs) * self.negative_ratio):
                key_a, key_b = rng.sample(group_keys, 2)
                mention_a = rng.choice(groups[key_a])
                mention_b = rng.choice(groups[key_b])
                pairs.append(
                    {
                        "entity_a": mention_a,
                        "entity_b": mention_b,
                        "label": 0,
                    }
                )

        rng.shuffle(pairs)
        return pairs
