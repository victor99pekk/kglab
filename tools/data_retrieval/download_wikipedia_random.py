#!/usr/bin/env python3
"""Download random Wikipedia articles from a public Hugging Face dataset.

Thin CLI wrapper around :func:`polygraph.data._wikipedia.download_wikipedia`.
For programmatic use, import ``polygraph.data.Data`` instead.

Usage::

    python tools/data_retrieval/download_wikipedia_random.py --count 50
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from polygraph.data._wikipedia import DEFAULT_MAX_SCAN, DEFAULT_SNAPSHOT, download_wikipedia

LOGGER = logging.getLogger("download_wikipedia")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download random Wikipedia articles from Hugging Face as Polygraph JSONL."
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of usable articles to download"
    )
    parser.add_argument("--language", default="en", help="Wikipedia language code (default: en)")
    parser.add_argument(
        "--snapshot",
        default=DEFAULT_SNAPSHOT,
        help="Hugging Face Wikimedia snapshot, for example 20231101",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wikipedia/random_articles.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--append", action="store_true", help="Append and skip existing IDs")
    parser.add_argument(
        "--min-chars", type=int, default=200, help="Skip articles shorter than this character count"
    )
    parser.add_argument("--seed", type=int, help="Reservoir-sampling seed")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=DEFAULT_MAX_SCAN,
        help="Maximum streamed rows to inspect",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    # Watchdog: PyArrow streaming threads can keep the process alive
    def _watchdog_force_exit(_signum: int, _frame: Any) -> None:
        print("Watchdog: forcing exit (background threads refused to stop)", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGALRM, _watchdog_force_exit)

    written = download_wikipedia(
        path=args.output,
        count=args.count,
        language=args.language,
        snapshot=args.snapshot,
        seed=args.seed,
        max_scan=args.max_scan,
        min_chars=args.min_chars,
        append=args.append,
    )
    print(f"Wrote {written} Wikipedia articles to {args.output}")

    signal.alarm(3)


if __name__ == "__main__":
    main()
