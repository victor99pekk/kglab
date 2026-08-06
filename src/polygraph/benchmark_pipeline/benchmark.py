"""Benchmark facade — namespace for all benchmark modules.

Usage::

    from polygraph.benchmark_pipeline import Benchmark
    from polygraph.pipelines import Baseline, Semantic
    from polygraph._shared.stage_config import PreprocessConfig, ResolutionConfig

    # Dedup benchmark
    result = Benchmark.Dedup(dataset="benchmarks/data/dedup_gold.jsonl").run(
        pipelines={
            "surface_minhash": Baseline(),
            "surface_layered": Baseline(
                preprocess=PreprocessConfig(doc_dedup_method="layered")
            ),
            "semantic": Semantic(),
        }
    )

    # Resolution benchmark
    result = Benchmark.Resolution(dataset="benchmarks/data/resolution_gold.jsonl").run(
        pipelines={
            "string": Baseline(),
            "embed": Baseline(resolution=ResolutionConfig(method="embedding")),
            "semantic": Semantic(),
        }
    )

    # Chunking benchmark
    result = Benchmark.Chunking(dataset="benchmarks/data/chunking_gold.jsonl").run(
        pipelines={
            "sentence": Baseline(preprocess=PreprocessConfig(chunk_method="sentence")),
            "semantic": Baseline(preprocess=PreprocessConfig(chunk_method="semantic")),
        }
    )

Each ``run(...)`` returns a ``StageResult`` — ``print(result)`` renders a
readable per-pipeline report.
"""

from polygraph.benchmark_pipeline.chunking import ChunkingRunner
from polygraph.benchmark_pipeline.dedup import DedupRunner
from polygraph.benchmark_pipeline.extraction import ExtractionRunner
from polygraph.benchmark_pipeline.quality import QualityFilterRunner
from polygraph.benchmark_pipeline.rag import RAGRunner
from polygraph.benchmark_pipeline.resolution import ResolutionRunner


class Benchmark:
    """Namespace for all benchmark modules.

    Each module tests one pipeline stage against a gold dataset.
    Every module takes pipeline instances so you control every
    hyperparameter at construction time.

    Get an overview of all benchmarks::

        print(Benchmark.help())

    Get details on a specific benchmark::

        print(Benchmark.Dedup.help())
        print(Benchmark.Extraction.help())
    """

    Dedup = DedupRunner
    """Deduplication quality — precision/recall against gold duplicate pairs."""

    Chunking = ChunkingRunner
    """Chunk boundary accuracy — F1 against gold chunk splits."""

    Resolution = ResolutionRunner
    """Entity merging quality — cluster F1 against gold entity groups."""

    Extraction = ExtractionRunner
    """NER quality — span-level precision/recall against gold annotations."""

    Quality = QualityFilterRunner
    """Garbage detection — accuracy against gold keep/reject labels."""

    RAG = RAGRunner
    """KG retrieval quality — entity recall, relation path accuracy, chunk retrieval for RAG."""

    @classmethod
    def help(cls) -> str:
        """Return an overview of all available benchmark modules.

        Each benchmark tests one stage of a KG pipeline against a gold
        dataset.  Use ``Benchmark.<Stage>.help()`` for details on a
        specific stage.

        Example::

            print(Benchmark.help())
            print(Benchmark.Dedup.help())
        """
        return (
            "Polygraph Benchmark Modules\n"
            "============================\n"
            "Six targeted benchmarks, each testing one pipeline stage\n"
            "against a gold dataset:\n\n"
            "  Benchmark.Dedup       — Document deduplication quality.\n"
            "                          Precision/recall/F1 against gold\n"
            "                          duplicate/non-duplicate pairs.\n\n"
            "  Benchmark.Chunking    — Chunk boundary accuracy.\n"
            "                          Boundary F1 against gold chunk splits.\n\n"
            "  Benchmark.Extraction  — Entity extraction (NER) quality.\n"
            "                          Span-level precision/recall/F1\n"
            "                          against gold entity annotations.\n\n"
            "  Benchmark.Resolution  — Entity resolution (merging) quality.\n"
            "                          Cluster F1 against gold entity groups.\n\n"
            "  Benchmark.Quality     — Content quality filter accuracy.\n"
            "                          Accuracy/precision/recall against\n"
            "                          gold keep/reject labels.\n\n"
            "  Benchmark.RAG         — KG retrieval quality for RAG.\n"
            "                          Entity recall@k, relation path accuracy,\n"
            "                          and chunk retrieval against gold QA pairs.\n\n"
            "Usage:\n"
            '  result = Benchmark.<Stage>(dataset="gold.jsonl").run(\n'
            '      pipelines={"label": MyPipeline()}\n'
            "  )\n\n"
            "For details on a specific benchmark, call its help() method:\n"
            "  print(Benchmark.Dedup.help())\n"
        )
