"""Polygraph — research pipeline for building high-quality KGs.

Usage:
    python main.py                                    # Baseline pipeline
    python main.py --input data/my_corpus/             # Custom input
    python main.py --variant graphgen                  # Different pipeline

Create custom pipelines in src/polygraph/pipelines/
"""

import argparse
from pathlib import Path

from polygraph.pipelines import Baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Polygraph — build and evaluate knowledge graphs")
    parser.add_argument(
        "--input", "-i", type=Path, nargs="+", default=[Path("data/debugg_sample/")]
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("output/baseline"))
    parser.add_argument(
        "--variant", default="baseline", choices=["baseline"], help="Pipeline variant to run"
    )
    args = parser.parse_args()

    pipeline = Baseline(
        input_paths=[str(p) for p in args.input],
        output_dir=str(args.output),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
