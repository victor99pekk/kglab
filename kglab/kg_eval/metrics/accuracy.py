"""Accuracy evaluation using an LLM judge.

Provides two metrics:
  - Sampled semantic accuracy: LLM judges a random sample of triples for
    factual correctness, with binomial confidence intervals.
  - Triple classification F1: LLM classifies real vs. corrupted triples;
    reports precision, recall, and F1.

The LLM client is injected — any callable ``(prompt: str) -> str`` works.
There is no hard dependency on a specific provider.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import networkx as nx

from kglab.kg_eval._base import BaseEvaluator

logger = logging.getLogger(__name__)


class AccuracyEvaluator(BaseEvaluator):
    """Evaluate KG accuracy using an LLM judge.

    The LLM client is injected via the constructor — any callable that takes
    a prompt string and returns a response string is accepted. This keeps the
    evaluator provider-agnostic (DeepSeek, OpenAI, local model, etc.).

    Usage:
        def my_llm(prompt: str) -> str:
            return openai.ChatCompletion.create(...)["choices"][0]["message"]["content"]

        evaluator = AccuracyEvaluator(llm_client=my_llm, sample_size=50)
        report = evaluator.evaluate(graph, entities, triples)
    """

    # ── Constants ──────────────────────────────────────────────

    # Z-score for 95% binomial confidence interval (Wilson score)
    _Z_95: float = 1.96

    _JUDGE_SYSTEM_PROMPT: str = (
        "You are a fact-checking assistant. Your task is to evaluate whether "
        "a knowledge graph triple states a factually correct claim.\n\n"
        "Respond with exactly one line: TRUE or FALSE, followed by a colon "
        "and a brief reason (one sentence).\n"
        "Example: TRUE: The claim is consistent with well-known facts.\n"
        "Example: FALSE: The claim contradicts available information."
    )

    def __init__(
        self,
        llm_client: Callable[[str], str],
        sample_size: int = 50,
        seed: int | None = 42,
    ) -> None:
        """Initialize the accuracy evaluator.

        Args:
            llm_client: A callable that takes a prompt string and returns
                a response string. Must be provided — there is no default.
            sample_size: Number of triples to sample for semantic accuracy
                evaluation (default 50).
            seed: Random seed for reproducible sampling.
        """
        if llm_client is None:
            raise ValueError("llm_client is required — inject your LLM provider.")
        self._llm = llm_client
        self.sample_size = sample_size
        self._rng = random.Random(seed)

    # ── BaseEvaluator contract ──

    def evaluate(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Run both accuracy metrics and return an aggregated report."""
        sem = self.semantic_accuracy(entities, triples)
        tcf = self.triple_classification(entities, triples)
        return {**sem, **tcf}

    # ── Semantic Accuracy ──────────────────────────────────────

    def semantic_accuracy(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Sample triples and ask the LLM judge whether each is factually correct.

        Returns accuracy with a 95% binomial confidence interval (Wilson score).
        """
        if not triples:
            return {
                "semantic_accuracy": 0.0,
                "semantic_confidence_95": [0.0, 0.0],
                "semantic_sample_size": 0,
                "semantic_judgments": [],
            }

        n_sample = min(self.sample_size, len(triples))
        sample = self._rng.sample(triples, n_sample)

        judgments: list[dict[str, Any]] = []
        correct = 0

        for subj, pred, obj, _source in sample:
            prompt = self._build_triple_prompt(subj, pred, obj, entities)
            try:
                response = self._llm(prompt)
            except Exception:
                logger.warning("LLM call failed for triple (%s, %s, %s)", subj, pred, obj)
                response = "ERROR: LLM call failed"

            is_correct, reason = self._parse_judgment(response)
            if is_correct:
                correct += 1

            judgments.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "correct": is_correct,
                    "reason": reason,
                }
            )

        accuracy = correct / n_sample
        ci_low, ci_high = self._wilson_ci(correct, n_sample)

        return {
            "semantic_accuracy": round(accuracy, 4),
            "semantic_confidence_95": [round(ci_low, 4), round(ci_high, 4)],
            "semantic_sample_size": n_sample,
            "semantic_correct_count": correct,
            "semantic_judgments": judgments,
        }

    # ── Triple Classification ──────────────────────────────────

    def triple_classification(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Classify real vs. corrupted triples and compute F1.

        Generates negative triples by swapping objects within the same
        predicate group. Uses the LLM judge to classify each as true/false.
        """
        if not triples:
            return {
                "triple_classification_f1": 0.0,
                "triple_classification_precision": 0.0,
                "triple_classification_recall": 0.0,
                "triple_classification_accuracy": 0.0,
            }

        # Generate negative triples
        negatives = self._generate_negatives(triples)
        n_neg = min(len(negatives), len(triples))
        negatives = self._rng.sample(negatives, n_neg)

        # Combine and shuffle
        all_items: list[tuple[tuple[str, str, str, str], bool]] = []
        for t in triples[:n_neg]:
            all_items.append((t, True))  # positive
        for t in negatives:
            all_items.append((t, False))  # negative
        self._rng.shuffle(all_items)

        # Classify each
        tp = fp = tn = fn = 0
        for triple, is_real in all_items:
            subj, pred, obj, _source = triple
            prompt = self._build_triple_prompt(subj, pred, obj, entities)
            try:
                response = self._llm(prompt)
            except Exception:
                logger.warning("LLM call failed for triple (%s, %s, %s)", subj, pred, obj)
                continue

            predicted_real, _reason = self._parse_judgment(response)

            if is_real and predicted_real:
                tp += 1
            elif is_real and not predicted_real:
                fn += 1
            elif not is_real and predicted_real:
                fp += 1
            else:
                tn += 1

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

        return {
            "triple_classification_f1": round(f1, 4),
            "triple_classification_precision": round(precision, 4),
            "triple_classification_recall": round(recall, 4),
            "triple_classification_accuracy": round(accuracy, 4),
            "triple_classification_confusion": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
        }

    # ── Helpers ─────────────────────────────────────────────────

    def _build_triple_prompt(
        self,
        subj: str,
        pred: str,
        obj: str,
        entities: list[dict[str, Any]],
    ) -> str:
        """Build a judge prompt for a single triple."""
        # Look up entity types for context
        entity_map = {e.get("name", ""): e for e in entities}
        subj_info = entity_map.get(subj, {})
        obj_info = entity_map.get(obj, {})

        subj_type = subj_info.get("type", "ENTITY")
        obj_type = obj_info.get("type", "ENTITY")
        subj_desc = subj_info.get("description", "")
        obj_desc = obj_info.get("description", "")

        parts = [
            self._JUDGE_SYSTEM_PROMPT,
            "",
            "Evaluate this knowledge graph triple:",
            f"  Subject: {subj} (type: {subj_type})",
            f"  Predicate: {pred}",
            f"  Object: {obj} (type: {obj_type})",
        ]
        if subj_desc:
            parts.append(f"  Subject description: {subj_desc}")
        if obj_desc:
            parts.append(f"  Object description: {obj_desc}")

        parts.append("")
        parts.append(
            "Is this statement factually correct? Answer TRUE or FALSE with a brief reason."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_judgment(response: str) -> tuple[bool, str]:
        """Parse an LLM judge response into (is_correct, reason)."""
        if not response:
            return False, "empty response"

        first_line = response.strip().split("\n")[0]
        upper = first_line.upper()

        if upper.startswith("TRUE"):
            return True, first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
        elif upper.startswith("FALSE"):
            return False, first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
        else:
            # Best-effort: look for keywords
            if "TRUE" in upper and "FALSE" not in upper:
                return True, first_line
            return False, first_line

    @staticmethod
    def _wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score confidence interval for a binomial proportion.

        Args:
            correct: Number of successes.
            total: Number of trials.
            z: Z-score (1.96 for 95% CI).

        Returns:
            (lower_bound, upper_bound) as floats in [0, 1].
        """
        if total == 0:
            return 0.0, 0.0

        p_hat = correct / total
        z2 = z * z
        denom = 1 + z2 / total
        center = (p_hat + z2 / (2 * total)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    def _generate_negatives(
        self,
        triples: list[tuple[str, str, str, str]],
    ) -> list[tuple[str, str, str, str]]:
        """Generate negative triples by swapping objects within predicate groups.

        Corrupts existing triples to create plausible false statements. For
        each predicate, swaps objects between triples that share the same
        predicate. Falls back to random entity substitution for singleton
        predicates.
        """
        if len(triples) < 2:
            return []

        # Group by predicate
        by_pred: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        for t in triples:
            by_pred[t[1]].append(t)

        negatives: list[tuple[str, str, str, str]] = []

        for pred, group in by_pred.items():
            if len(group) < 2:
                # Singleton predicate — swap object with a random entity
                subj, _, obj, src = group[0]
                all_objs = [t[2] for t in triples if t[2] != obj]
                if all_objs:
                    new_obj = self._rng.choice(all_objs)
                    negatives.append((subj, pred, new_obj, src))
            else:
                # Swap objects within the group
                for i, (subj, _, obj, src) in enumerate(group):
                    candidates = [t[2] for j, t in enumerate(group) if j != i and t[2] != obj]
                    if candidates:
                        new_obj = self._rng.choice(candidates)
                        negatives.append((subj, pred, new_obj, src))

        return negatives
