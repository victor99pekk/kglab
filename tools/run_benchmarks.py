#!/usr/bin/env python3
"""Run benchmark stages across the supported pipeline variants.

Usage:
    uv run python tools/run_benchmarks.py                   # all implemented benchmarks
    uv run python tools/run_benchmarks.py --stage dedup     # specific stages
    uv run python tools/run_benchmarks.py --output benchmarks/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kglab._shared.stage_config import ResolutionConfig
from kglab.benchmark_pipeline import Benchmark, StageResult, format_stages
from kglab.pipelines import Baseline, Semantic

ALL_STAGES = ["dedup", "chunking", "extraction", "resolution", "quality", "rag"]


def _pipelines_for(stage: str) -> dict:
    """Build the pipeline variants to benchmark for a given stage."""
    surface = Baseline()
    semantic = Semantic()

    if stage == "resolution":
        return {
            "string": surface,
            "embed": Baseline(resolution=ResolutionConfig(method="embedding")),
            "semantic": semantic,
        }
    return {"surface": surface, "semantic": semantic}


def _runner_attr(stage: str) -> str:
    """Map stage name to Benchmark attribute (RAG is all-caps)."""
    return "RAG" if stage == "rag" else stage.capitalize()


def _run_stage(
    stage: str,
    output_dir: Path,
    input_paths: list[str] | None = None,
    dataset: Path | None = None,
) -> StageResult:
    runner_cls = getattr(Benchmark, _runner_attr(stage))
    pipelines = _pipelines_for(stage)

    print("=" * 70)
    print(f"Benchmark: {stage}")
    print(f"Pipelines: {list(pipelines.keys())}")
    print("=" * 70)

    runner = runner_cls(dataset=dataset) if dataset is not None else runner_cls()
    if stage == "rag":
        # RAG builds a KG from input documents — it needs a corpus, unlike
        # the gold-only stages. Pass --input to provide it.
        results = runner.run(pipelines=pipelines, input_paths=input_paths)
    else:
        results = runner.run(pipelines=pipelines)

    out = output_dir / f"{stage}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results.to_dict(), indent=2, default=str))
    print(results)
    print(f"  → {out}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KGLab benchmark stages across pipeline variants.",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=ALL_STAGES,
        default=None,
        help="Benchmark stages to run (default: all implemented).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results"),
        help="Where to write per-stage result JSON.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Input files/dirs for the RAG stage (it builds a KG per pipeline "
        "from these documents). Other stages score against gold datasets only.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Gold dataset JSONL for the stage(s) run, e.g. a manually placed "
        "rag_gold.jsonl or quality_gold.jsonl (overrides the built-in gold).",
    )
    args = parser.parse_args()

    selected = args.stage or ALL_STAGES
    summary: dict = {}
    datasets: dict[str, Path | None] = {}

    for stage in selected:
        # Fail loudly: a broken stage/pipeline must crash the run so you know.
        result = _run_stage(stage, args.output, args.input, args.dataset)
        summary[stage] = result.to_dict()
        datasets[stage] = result.dataset

    if len(selected) > 1:
        # Each stage already rendered its own table above; the combined report
        # is only added for multi-stage runs so the output stays readable.
        print(format_stages(summary, datasets=datasets))

    summary_path = args.output / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Combined summary → {summary_path}")


if __name__ == "__main__":
    main()
