"""Chunking benchmark — compare chunk boundary accuracy against gold splits.

Each pipeline chunks the gold texts and boundary F1 is computed against
gold chunk boundaries.

Gold dataset format (JSONL)::

    {"text": "...", "chunks": [{"name": "EU", "type": "ORG"}, ...]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polygraph.pipelines import Pipeline

#: Default location for the gold chunking dataset when no ``dataset`` is passed.
DEFAULT_CHUNKING_DATASET = "benchmarks/data/chunking_gold.jsonl"

#: Human-readable description of the gold dataset schema.
DATASET_FORMAT = """\
Each line of the gold JSONL is one object:
  "text"   — raw text to chunk.
  "chunks" — gold entities that must stay intact within a single chunk,
             as a list of {"name": str, "type": str}.
"""


class ChunkingRunner:
    """Benchmark chunking quality across pipeline instances.

    Compares the chunk boundaries produced by each pipeline against
    a gold-standard segmentation.  Useful for evaluating different
    chunking strategies (sentence, semantic, fixed-size, etc.).

    Primary metric: boundary F1 (how well chunk boundaries align with gold).
    """

    def __init__(self, dataset: str | Path | None = None) -> None:
        self.dataset = Path(dataset) if dataset is not None else None

    @classmethod
    def help(cls) -> str:
        """Return a human-readable description of this benchmark.

        Use this to understand what the benchmark measures, the gold
        dataset format, and the metrics it produces — without reading
        the source code.

        Example::

            print(ChunkingRunner.help())
        """
        return (
            "Chunking Benchmark\n"
            "==================\n"
            "What it tests:\n"
            "  Whether a pipeline splits text into chunks that align with\n"
            "  a gold-standard segmentation. Different chunking strategies\n"
            "  (sentence, semantic, fixed-size) can dramatically affect\n"
            "  downstream entity extraction and relation quality.\n\n"
            "Why it matters:\n"
            "  Chunk boundaries determine what context an entity extractor\n"
            "  sees. Poor chunking can split entities across chunks or\n"
            "  merge unrelated concepts, degrading KG quality.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"text": "...", "chunks": ["chunk 1", "chunk 2", ...]}\n\n'
            "Primary metric:\n"
            "  Boundary F1 — harmonic mean of precision and recall of\n"
            "  chunk boundaries against gold boundaries.\n"
        )

    def run(self, pipelines: dict[str, Pipeline]) -> dict[str, Any]:
        """Run chunking benchmark and return metrics per pipeline."""
        raise NotImplementedError("Chunking benchmark not yet implemented")
