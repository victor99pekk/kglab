"""Polygraph — research pipeline for building high-quality KGs.

Usage:
    # Run an experiment from a YAML config (recommended)
    python main.py --experiment experiments/kg/001_baseline/config.yaml

    # Direct pipeline usage (backward compatible)
    python main.py                                              # Baseline pipeline
    python main.py --variant baseline --input data/my_corpus/   # Custom input
    python main.py --neo4j                                      # Upload to Neo4j
    python main.py --neo4j --clear-neo4j                        # Wipe then upload
    python main.py --ontology configs/my_ontology.yaml          # Custom ontology

Create custom pipelines in src/polygraph/pipelines/
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from polygraph.pipelines import PIPELINE_REGISTRY

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polygraph — build and evaluate knowledge graphs",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=Path,
        default=None,
        help="Path to an experiment YAML config file. When provided, runs the "
        "full benchmark pipeline and writes a results_summary.json.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        default=[Path("data/debugg_sample/")],
        help="Input data paths (ignored when --experiment is used).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/baseline"),
        help="Output directory (ignored when --experiment is used).",
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=list(PIPELINE_REGISTRY.keys()),
        help="Pipeline variant to run (ignored when --experiment is used).",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Path to ontology YAML (default: configs/default_ontology.yaml).",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        default=False,
        help="Upload the exported KG to Neo4j after the pipeline completes.",
    )
    parser.add_argument(
        "--clear-neo4j",
        action="store_true",
        default=False,
        help="Wipe the Neo4j database before uploading (requires --neo4j).",
    )
    args = parser.parse_args()

    # ── Experiment mode (YAML config) ──────────────────────────
    if args.experiment:
        from polygraph.benchmark_pipeline import BenchmarkRunner
        from polygraph.benchmark_pipeline.config import ExperimentConfig

        config = ExperimentConfig.from_yaml(args.experiment)
        runner = BenchmarkRunner(config)
        runner.run()
        return

    # ── Direct pipeline mode (backward compatible) ─────────────
    pipeline_cls = PIPELINE_REGISTRY[args.variant]
    pipeline_kwargs: dict = {}
    if args.ontology:
        pipeline_kwargs["ontology_path"] = str(args.ontology)

    pipeline = pipeline_cls(
        input_paths=[str(p) for p in args.input],
        output_dir=str(args.output),
        **pipeline_kwargs,
    )
    pipeline.execute()

    if args.neo4j:
        pipeline.upload_to_neo4j(clear=args.clear_neo4j)


if __name__ == "__main__":
    main()
