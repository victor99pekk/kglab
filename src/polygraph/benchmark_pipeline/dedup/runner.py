"""Deduplication benchmark — compare pipeline dedup quality against gold pairs.

Each pipeline is run on the gold dataset and its dedup decisions are scored
against known duplicate / non-duplicate labels.

Gold dataset format (JSONL)::

    {"text_a": "...", "text_b": "...", "label": "duplicate"}
    {"text_a": "...", "text_b": "...", "label": "not_duplicate"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polygraph.pipelines import Pipeline


class DedupRunner:
    """Benchmark deduplication quality across pipeline instances.

    Each pipeline's dedup configuration (method, threshold) is whatever
    was set at construction time — the benchmark just runs and scores.

    Compares dedup decisions against gold duplicate/non-duplicate labels.
    """

    def __init__(self, dataset: str | Path) -> None:
        self.dataset = Path(dataset)

    @classmethod
    def help(cls) -> str:
        """Return a human-readable description of this benchmark.

        Use this to understand what the benchmark measures, the gold
        dataset format, and the metrics it produces — without reading
        the source code.

        Example::

            print(DedupRunner.help())
        """
        return (
            "Document Deduplication Benchmark\n"
            "================================\n"
            "What it tests:\n"
            "  Whether a pipeline correctly identifies duplicate and\n"
            "  near-duplicate documents. Each pipeline's dedup method\n"
            "  (e.g., MinHash, semantic embeddings) is scored against\n"
            "  gold duplicate/non-duplicate pair labels.\n\n"
            "Why it matters:\n"
            "  Duplicate documents inflate entity counts and can bias\n"
            "  relation extraction. A good dedup step keeps the KG\n"
            "  compact and accurate without losing unique information.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"text_a": "...", "text_b": "...", "label": "duplicate"}\n'
            '  {"text_a": "...", "text_b": "...", "label": "not_duplicate"}\n\n'
            "Primary metrics:\n"
            "  Precision — fraction of predicted duplicates that are true duplicates.\n"
            "  Recall — fraction of true duplicates that were found.\n"
            "  F1 — harmonic mean of precision and recall.\n"
        )

    def run(self, pipelines: dict[str, Pipeline]) -> dict[str, Any]:
        """Run dedup benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.  Each pipeline should be
                fully configured (dedup method, threshold, etc.) but does
                not need ``input_paths`` or ``output_dir``.
        """
        # TODO: implement
        # 1. Load gold pairs from self.dataset
        # 2. For each pipeline, extract its dedup method/threshold
        #    from pipeline._preprocessor.config
        # 3. Run dedup on the gold texts
        # 4. Compare cluster assignments against gold labels
        # 5. Compute precision / recall / F1 per pipeline
        raise NotImplementedError("Dedup benchmark not yet implemented")
