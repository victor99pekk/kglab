"""Polygraph — research pipeline for building high-quality KGs.

Usage:
    # Run an experiment from a YAML config (recommended)
    python main.py --experiment experiments/kg/001_baseline/config.yaml

    # Direct pipeline usage (backward compatible)
    python main.py                                              # Baseline pipeline
    python main.py --variant baseline --input data/my_corpus/   # Custom input
    python main.py --neo4j                                      # Stream directly into Neo4j
    python main.py --neo4j --clear-neo4j                        # Wipe Neo4j then stream
    python main.py --ontology configs/my_ontology.yaml          # Custom ontology

Create custom pipelines in src/polygraph/pipelines/
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from polygraph._shared.run_manifest import next_run_dir
from polygraph._shared.stage_config import LinkingConfig
from polygraph.kg_eval import evaluate_kg
from polygraph.pipelines import PIPELINE_REGISTRY

load_dotenv()


def _resolve_output_dir(output_arg: Path | None) -> Path:
    """Resolve the output directory for a pipeline run.

    If ``--output`` is explicitly provided, use it as-is (backward compatible).
    Otherwise, auto-increment under ``generated_KGs/KG_0``, ``generated_KGs/KG_1``, …
    """
    if output_arg is not None:
        return output_arg.resolve()
    return next_run_dir("generated_KGs").resolve()


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
        default=None,
        help="Output directory.  When omitted, auto-increments under "
        "generated_KGs/ (KG_0, KG_1, …).  Ignored when --experiment is used.",
    )
    parser.add_argument(
        "--variant",
        default="surface",
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
        help="Stream the KG directly into Neo4j while building (no in-memory graph).",
    )
    parser.add_argument(
        "--clear-neo4j",
        action="store_true",
        default=False,
        help="Wipe the Neo4j database before streaming (requires --neo4j).",
    )
    parser.add_argument(
        "--linking",
        action="store_true",
        default=False,
        help="Link extracted entities to Wikidata after resolution.",
    )
    args = parser.parse_args()

    # ── Experiment mode (YAML config) ──────────────────────────
    if args.experiment:
        from polygraph.benchmark_pipeline import BenchmarkRunner
        from polygraph.benchmark_pipeline.config import ExperimentConfig

        config = ExperimentConfig.from_yaml(args.experiment)
        runner = BenchmarkRunner(config)
        runner.run()

        if args.neo4j:
            runner.pipeline.upload_to_neo4j(clear=args.clear_neo4j)
        return

    # ── Direct pipeline mode (backward compatible) ─────────────
    pipeline_cls = PIPELINE_REGISTRY[args.variant]
    pipeline_kwargs: dict = {}
    if args.ontology:
        pipeline_kwargs["ontology_path"] = str(args.ontology)
    if args.neo4j:
        # Stream the KG directly into Neo4j during the build — no in-memory
        # graph, and the returned graph_store is a live Neo4jGraphStore.
        pipeline_kwargs["graph_store_backend"] = "neo4j"
        pipeline_kwargs["graph_store_options"] = {"clear": args.clear_neo4j}

    output_dir = _resolve_output_dir(args.output)
    pipeline = pipeline_cls(
        linking=LinkingConfig(enabled=args.linking),
        **pipeline_kwargs,
    )
    kg = pipeline.execute(
        input_paths=[str(p) for p in args.input],
        output_dir=str(output_dir),
    )

    # Evaluation is external to the pipeline — score the built KG here.
    evaluate_kg(kg, output_dir=output_dir)


if __name__ == "__main__":
    main()
