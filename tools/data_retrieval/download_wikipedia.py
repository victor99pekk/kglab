#!/usr/bin/env python3
"""Download random Wikipedia articles from a public Hugging Face dataset.

Uses ``wikimedia/wikipedia`` in streaming mode. Generated records contain
Polygraph's required fields (id, text, title, url) plus dataset provenance.
The Hugging Face dataset already removes references and other unwanted
sections from article text.

This repository is MIT-licensed. Wikipedia content remains subject to the
licenses and attribution requirements of the relevant Wikimedia project.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

LOGGER = logging.getLogger("download_wikipedia")
DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_SNAPSHOT = "20231101"
DEFAULT_MAX_SCAN = 10_000
DATASET_LICENSE = "CC BY-SA 3.0 / GFDL"
DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"


class HuggingFaceDownloadError(RuntimeError):
    """Raised when the public Hugging Face dataset cannot be read."""


class HuggingFaceWikipediaClient:
    """Stream one Wikimedia Wikipedia language split and reservoir-sample rows."""

    def __init__(
        self,
        language: str = "en",
        snapshot: str = DEFAULT_SNAPSHOT,
        seed: int | None = None,
        max_scan: int = DEFAULT_MAX_SCAN,
    ) -> None:
        if not language.isalpha() or not 2 <= len(language) <= 12:
            raise ValueError("language must be a Wikipedia language code, for example 'en' or 'vi'")
        if not snapshot.isdigit():
            raise ValueError("snapshot must be a date-like dataset version, for example '20231101'")
        if max_scan < 1:
            raise ValueError("max_scan must be positive")

        self.language = language.lower()
        self.snapshot = snapshot
        self.config = f"{snapshot}.{self.language}"
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**32)
        self.max_scan = max_scan

    def fetch_random(
        self,
        limit: int,
        min_chars: int = 200,
        excluded_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Stream rows and reservoir-sample until ``limit`` usable articles are collected."""
        if limit < 1:
            return []

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise HuggingFaceDownloadError(
                "Missing Hugging Face Datasets. Install with: uv sync --extra data"
            ) from exc

        dataset = None
        try:
            dataset = load_dataset(
                DEFAULT_DATASET,
                self.config,
                split="train",
                streaming=True,
            )
            rows: Iterable[dict[str, Any]] = dataset
            records: list[dict[str, Any]] = []
            excluded = excluded_ids or set()
            eligible_seen = 0
            rng = random.Random(self.seed)
            for scanned, row in enumerate(rows, start=1):
                record = self.row_to_record(
                    row,
                    language=self.language,
                    snapshot=self.snapshot,
                    downloaded_at=utc_now(),
                    min_chars=min_chars,
                )
                if record is not None and record["id"] not in excluded:
                    eligible_seen += 1
                    if len(records) < limit:
                        records.append(record)
                    else:
                        replacement = rng.randrange(eligible_seen)
                        if replacement < limit:
                            records[replacement] = record
                if scanned >= self.max_scan:
                    break
        except HuggingFaceDownloadError:
            raise
        except Exception as exc:
            raise HuggingFaceDownloadError(
                f"could not stream {DEFAULT_DATASET}/{self.config}: {exc}"
            ) from exc
        finally:
            if dataset is not None:
                _close_dataset(dataset)

        if len(records) >= limit:
            return records

        raise HuggingFaceDownloadError(
            f"found {len(records)} usable articles after scanning {self.max_scan} rows; "
            f"wanted {limit}. Lower --min-chars or raise --max-scan."
        )

    @staticmethod
    def row_to_record(
        row: dict[str, Any],
        language: str,
        snapshot: str,
        downloaded_at: str,
        min_chars: int = 200,
    ) -> dict[str, Any] | None:
        """Convert one Hugging Face row to the Polygraph JSONL schema."""
        raw_id = row.get("id")
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        if raw_id is None or not title or not text or len(text) < min_chars:
            return None

        page_id = str(raw_id)
        url = str(row.get("url", "")).strip() or _article_url(language, title)
        return {
            "id": f"wikipedia:{language}:{page_id}",
            "text": text,
            "title": title,
            "url": url,
            "source": f"huggingface:{DEFAULT_DATASET}",
            "dataset": DEFAULT_DATASET,
            "dataset_config": f"{snapshot}.{language}",
            "language": language,
            "page_id": page_id,
            "downloaded_at": downloaded_at,
            "license": DATASET_LICENSE,
            "license_url": DATASET_URL,
            "categories": [],
            "metadata_note": "HF Wikimedia Wikipedia rows do not include Wikipedia categories.",
        }


def download_articles(
    client: HuggingFaceWikipediaClient,
    count: int,
    output: Path,
    append: bool = False,
    min_chars: int = 200,
) -> int:
    """Download ``count`` new records and write them to ``output``."""
    if count < 1:
        raise ValueError("count must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_ids(output) if append and output.exists() else set()
    candidates = client.fetch_random(count, min_chars=min_chars, excluded_ids=existing_ids)
    records = []
    for record in candidates:
        if record["id"] not in existing_ids:
            existing_ids.add(record["id"])
            records.append(record)
        if len(records) == count:
            break
    if len(records) < count:
        raise HuggingFaceDownloadError(
            f"downloaded {len(records)} new articles; wanted {count}. "
            "Increase --max-scan or use a different --seed."
        )

    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


def _existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if value.get("id"):
                ids.add(str(value["id"]))
    return ids


def _close_dataset(dataset: Any) -> None:
    """Shut down background workers of a streaming Hugging Face dataset.

    Without this, PyArrow/``datasets`` streaming threads can keep the
    process alive after the iteration loop exits.  We try every available
    teardown path and then force a garbage collection so any weakly-held
    Arrow resources are released.
    """
    # 1. Cancel any in-flight shard downloads
    for attr in ("_ex_iterable", "_iterable"):
        it = getattr(dataset, attr, None)
        if it is not None:
            for method in ("close", "cancel", "shutdown"):
                closer = getattr(it, method, None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
            try:
                setattr(dataset, attr, None)
            except Exception:
                pass

    # 2. Call any public cleanup hook on the dataset itself
    for method in ("cleanup_cache_files", "close", "_cleanup"):
        closer = getattr(dataset, method, None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    # 3. Force garbage collection to release Arrow file handles
    gc.collect()


def _article_url(language: str, title: str) -> str:
    slug = quote(title.replace(" ", "_"), safe="()!,:;@&=+$-_.~'")
    return f"https://{language}.wikipedia.org/wiki/{slug}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download random Wikipedia articles from Hugging Face as Polygraph JSONL."
    )
    parser.add_argument("--count", type=int, default=20, help="Number of usable articles to download")
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

    # -- watchdog: PyArrow streaming threads can keep the process alive  --
    # after the download loop finishes.  Register a SIGALRM handler now
    # but only arm it *after* the real work is done, so it acts as a
    # grace-period timeout during shutdown rather than cutting off the
    # download itself.
    def _watchdog_force_exit(_signum: int, _frame: Any) -> None:
        print("Watchdog: forcing exit (background threads refused to stop)", file=sys.stderr)
        os._exit(0)

    signal.signal(signal.SIGALRM, _watchdog_force_exit)

    client = HuggingFaceWikipediaClient(
        language=args.language,
        snapshot=args.snapshot,
        seed=args.seed,
        max_scan=args.max_scan,
    )
    written = download_articles(
        client,
        count=args.count,
        output=args.output,
        append=args.append,
        min_chars=args.min_chars,
    )
    print(f"Wrote {written} Wikipedia articles to {args.output}")

    # Work is done — give the process 3 seconds to exit cleanly.  If
    # PyArrow workers keep it alive beyond that, SIGALRM force-kills it.
    signal.alarm(3)


if __name__ == "__main__":
    main()
