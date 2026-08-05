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

from polygraph._shared.stage_config import PreprocessConfig, ResolutionConfig
from polygraph.benchmark_pipeline import Benchmark
from polygraph.pipelines import Baseline, Semantic

ALL_STAGES = ["dedup", "chunking", "extraction", "resolution", "quality", "rag"]


def _pipelines_for(stage: str) -> dict:
    """Build the pipeline variants to benchmark for a given stage."""
    surface = Baseline()
    semantic = Semantic()

    if stage == "dedup":
        return {
            "surface": surface,
            "exact": Baseline(preprocess=PreprocessConfig(doc_dedup_method="exact")),
            "minhash": Baseline(preprocess=PreprocessConfig(doc_dedup_method="minhash")),
            "semantic": semantic,
        }
    if stage == "resolution":
        return {
            "string": surface,
            "embed": Baseline(resolution=ResolutionConfig(method="embedding")),
            "semantic": semantic,
        }
    if stage == "chunking":
        return {
            "sentence": Baseline(preprocess=PreprocessConfig(chunk_method="sentence")),
            "surface": surface,
            "semantic": semantic,
        }
    return {"surface": surface, "semantic": semantic}


def _format(metrics: dict) -> str:
    f1 = metrics.get("f1")
    suffix = f" F1={f1:.4f}" if f1 is not None else ""
    return f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}{suffix}"


def _runner_attr(stage: str) -> str:
    """Map stage name to Benchmark attribute (RAG is all-caps)."""
    return "RAG" if stage == "rag" else stage.capitalize()


def _run_stage(stage: str, output_dir: Path) -> dict:
    runner_cls = getattr(Benchmark, _runner_attr(stage))
    pipelines = _pipelines_for(stage)

    print("=" * 70)
    print(f"Benchmark: {stage}")
    print(f"Pipelines: {list(pipelines.keys())}")
    print("=" * 70)

    runner = runner_cls()
    results = runner.run(pipelines=pipelines)

    for name, metrics in results.items():
        print(f"  {name:<12} {_format(metrics)}")

    out = output_dir / f"{stage}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"  → {out}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Polygraph benchmark stages across pipeline variants.",
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
    args = parser.parse_args()

    selected = args.stage or ALL_STAGES
    summary: dict = {}

    for stage in selected:
        try:
            summary[stage] = _run_stage(stage, args.output)
        except NotImplementedError:
            print(f"[skip] {stage} — runner not yet implemented.\n")
        except Exception as exc:
            print(f"[skip] {stage} — {type(exc).__name__}: {exc}\n")

    summary_path = args.output / "benchmark_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Combined summary → {summary_path}")


if __name__ == "__main__":
    main()
