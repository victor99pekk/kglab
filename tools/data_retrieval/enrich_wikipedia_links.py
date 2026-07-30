#!/usr/bin/env python3
"""Enrich a Polygraph Wikipedia JSONL file with inter-article hyperlinks.

Thin CLI wrapper around :func:`polygraph.data._wikipedia.enrich_wikipedia`.
For programmatic use, import ``polygraph.data.Data`` instead.

Usage::

    python tools/data_retrieval/enrich_wikipedia_links.py \\
        --input data/wikipedia/random_articles.jsonl \\
        --output data/wikipedia/random_articles_enriched.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from polygraph.data._wikipedia import REQUEST_DELAY, enrich_wikipedia

logger = logging.getLogger("enrich_wikipedia_links")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich Wikipedia JSONL records with outgoing hyperlinks via the Wikipedia API."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to existing Polygraph Wikipedia JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the enriched JSONL output.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Wikipedia language code (default: en).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds between API requests (default: {REQUEST_DELAY}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    enriched = enrich_wikipedia(
        input_path=args.input,
        output_path=args.output,
        language=args.language,
        delay=args.delay,
    )
    print(f"Done. {enriched} records enriched with outgoing links.")


if __name__ == "__main__":
    main()
