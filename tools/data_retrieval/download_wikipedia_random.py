#!/usr/bin/env python3
"""Download Wikipedia articles — random, specific URLs, or degree-targeted.

Thin CLI wrapper around :func:`kglab.data._wikipedia.download_wikipedia`.
For programmatic use, import ``kglab.data.Data`` instead.

Usage::

    # Random articles (default)
    python tools/data_retrieval/download_wikipedia_random.py --count 50

    # Specific articles from URLs
    python tools/data_retrieval/download_wikipedia_random.py \\
        --strategy specific --urls "https://en.wikipedia.org/wiki/Alan_Turing" ...

    # Degree-targeted
    python tools/data_retrieval/download_wikipedia_random.py \\
        --strategy degree --count 50 --target-degree 5.0
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from kglab.data._wikipedia import (
    DEFAULT_MAX_SCAN,
    DEFAULT_SNAPSHOT,
    DegreeSampler,
    RandomSampler,
    SpecificSampler,
    download_wikipedia,
)

LOGGER = logging.getLogger("download_wikipedia")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Wikipedia articles as KGLab JSONL.")
    parser.add_argument("--count", type=int, default=20, help="Number of articles to download")
    parser.add_argument("--language", default="en", help="Wikipedia language code (default: en)")
    parser.add_argument(
        "--snapshot",
        default=DEFAULT_SNAPSHOT,
        help="Hugging Face Wikimedia snapshot (for random strategy)",
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
    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "specific", "degree"],
        help="Download strategy (default: random)",
    )
    parser.add_argument(
        "--urls", nargs="*", help="Wikipedia article URLs (for 'specific' strategy)"
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Path to a text file with one URL per line (for 'specific' strategy)",
    )
    parser.add_argument(
        "--target-degree",
        type=float,
        default=3.0,
        help="Target average hyperlink degree (for 'degree' strategy)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        help="Hard cap on total articles (for 'degree' strategy; default: count * 5)",
    )
    parser.add_argument(
        "--exclude-namespaces",
        nargs="*",
        default=None,
        help="Namespace prefixes to skip (for 'degree' strategy), e.g. Help: Template:",
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

    if args.strategy == "specific":
        sampler = SpecificSampler(urls=args.urls, url_file=args.url_file)
    elif args.strategy == "degree":
        sampler = DegreeSampler(
            count=args.count,
            target_degree=args.target_degree,
            max_articles=args.max_articles,
            exclude_namespaces=args.exclude_namespaces,
            seed=args.seed,
            max_scan=args.max_scan,
            min_chars=args.min_chars,
        )
    else:
        sampler = RandomSampler(
            count=args.count,
            seed=args.seed,
            max_scan=args.max_scan,
            min_chars=args.min_chars,
        )

    written = download_wikipedia(
        path=args.output,
        sampler=sampler,
        language=args.language,
        snapshot=args.snapshot,
        append=args.append,
    )
    print(f"Wrote {written} Wikipedia articles to {args.output}")

    signal.alarm(3)


if __name__ == "__main__":
    main()
