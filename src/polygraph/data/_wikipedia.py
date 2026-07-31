"""Wikipedia data — download and enrich Polygraph JSONL from HuggingFace + Wikipedia API.

Provides two public functions:

* ``download_wikipedia()`` — stream random articles from ``wikimedia/wikipedia``
* ``enrich_wikipedia()`` — add outgoing hyperlinks via Wikipedia API

Both write Polygraph-compliant JSONL.  The enrichment step populates a
``links`` field that the preprocess ``normalize_links`` →
``HyperlinkExtractor`` pipeline uses to create ``hyperlinks_to`` edges.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import random
import re
import signal
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ── Download constants ────────────────────────────────────────

DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_SNAPSHOT = "20231101"
DEFAULT_MAX_SCAN = 10_000
DATASET_LICENSE = "CC BY-SA 3.0 / GFDL"
DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"

# ── Enrich constants ──────────────────────────────────────────

REQUEST_DELAY = 0.1  # seconds between Wikipedia API calls

# Wikipedia namespace prefixes to skip during degree expansion.
# These are meta/administrative pages, not encyclopedic articles.
_DEFAULT_EXCLUDE_NAMESPACES = ["Help:", "Template:"]


# ═══════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════


class HuggingFaceDownloadError(RuntimeError):
    """Raised when the public Hugging Face dataset cannot be read."""


# ═══════════════════════════════════════════════════════════════
# Download
# ═══════════════════════════════════════════════════════════════


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
                    downloaded_at=_utc_now(),
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


def download_wikipedia(
    path: str | Path,
    count: int = 20,
    language: str = "en",
    snapshot: str = DEFAULT_SNAPSHOT,
    seed: int | None = None,
    max_scan: int = DEFAULT_MAX_SCAN,
    min_chars: int = 200,
    append: bool = False,
    strategy: str = "random",
    urls: list[str] | None = None,
    url_file: str | Path | None = None,
    target_degree: float = 3.0,
    max_articles: int | None = None,
    exclude_namespaces: list[str] | None = None,
) -> int:
    """Download Wikipedia articles and write Polygraph JSONL.

    Supports three strategies:

    * ``"random"`` — reservoir-sample from HuggingFace (default).
    * ``"specific"`` — fetch articles from a list of URLs or a file of URLs.
    * ``"degree"`` — grow a connected set of articles to hit a target
      average hyperlink degree.

    Args:
        path: Output JSONL file path.
        count: Number of articles to download.
        language: Wikipedia language code (e.g., ``"en"``).
        snapshot: HuggingFace dataset snapshot (for ``"random"`` strategy).
        seed: Reservoir-sampling seed (random if ``None``).
        max_scan: Maximum streamed rows to inspect.
        min_chars: Skip articles shorter than this character count.
        append: Append to existing file (skip already-present IDs).
        strategy: ``"random"``, ``"specific"``, or ``"degree"``.
        urls: List of Wikipedia article URLs (required for ``"specific"``).
        url_file: Path to a text file with one URL per line (``"specific"``).
        target_degree: Target average hyperlink degree (``"degree"`` strategy).
        max_articles: Hard cap on total articles for degree strategy
            (defaults to ``count * 5``).
        exclude_namespaces: Wikipedia namespace prefixes to skip during
            degree expansion (e.g., ``["Help:", "Template:"]``). Defaults to
            ``["Help:", "Wikipedia:", "Template:", "File:", "Category:",
            "Portal:"]``. Pass an empty list to include all pages.

    Returns:
        Number of records written.
    """
    if strategy == "random":
        return _download_random(
            path=path,
            count=count,
            language=language,
            snapshot=snapshot,
            seed=seed,
            max_scan=max_scan,
            min_chars=min_chars,
            append=append,
        )
    elif strategy == "specific":
        return _download_specific(
            path=path,
            urls=urls,
            url_file=url_file,
            language=language,
            append=append,
        )
    elif strategy == "degree":
        return _download_degree_targeted(
            path=path,
            count=count,
            language=language,
            snapshot=snapshot,
            seed=seed,
            max_scan=max_scan,
            min_chars=min_chars,
            target_degree=target_degree,
            max_articles=max_articles,
            exclude_namespaces=exclude_namespaces,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: random, specific, degree")


def _download_random(
    path: str | Path,
    count: int,
    language: str,
    snapshot: str,
    seed: int | None,
    max_scan: int,
    min_chars: int,
    append: bool,
) -> int:
    """Reservoir-sample random articles from HuggingFace (original behaviour)."""
    if count < 1:
        raise ValueError("count must be positive")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_ids(output) if append and output.exists() else set()

    client = HuggingFaceWikipediaClient(
        language=language,
        snapshot=snapshot,
        seed=seed,
        max_scan=max_scan,
    )
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

    logger.info("Wrote %d Wikipedia articles to %s", len(records), output)

    # PyArrow streaming threads from HuggingFace ``datasets`` can keep the
    # process alive after the download loop exits.  Schedule a graceful
    # SIGALRM that force-exits after a short grace period so scripts and
    # notebooks terminate cleanly.
    _schedule_watchdog()

    return len(records)


_WIKI_URL_RE = re.compile(r"https?://([a-z]{2,12})\.wikipedia\.org/wiki/(.+)", re.IGNORECASE)


def _parse_wiki_url(url: str) -> tuple[str, str] | None:
    """Extract (language, title) from a Wikipedia URL.  Returns ``None`` on failure."""
    m = _WIKI_URL_RE.match(url.strip())
    if not m:
        return None
    lang = m.group(1).lower()
    title = m.group(2).replace("_", " ").strip()
    return lang, title


def _fetch_article_text(title: str, language: str = "en") -> dict[str, Any] | None:
    """Fetch article text + metadata from the Wikipedia API.

    Uses ``action=query&prop=extracts|info&exintro=0&explaintext``.
    Returns a dict with ``page_id``, ``title``, ``text``, ``url`` or ``None``.
    """
    import requests

    api_url = f"https://{language}.wikipedia.org/w/api.php"
    params: dict[str, Any] = {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": 1,
        "exintro": 0,
        "inprop": "url",
        "format": "json",
    }
    headers = {
        "User-Agent": "Polygraph/0.2 (https://github.com/user/polygraph; research KG construction)"
    }

    try:
        resp = requests.get(api_url, params=params, timeout=30, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("API error fetching article '%s': %s", title, exc)
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        logger.warning("Article not found: %s", title)
        return None

    return {
        "page_id": str(page.get("pageid", "")),
        "title": page.get("title", title),
        "text": page.get("extract", ""),
        "url": page.get("fullurl", page.get("canonicalurl", "")),
    }


def _download_specific(
    path: str | Path,
    urls: list[str] | None,
    url_file: str | Path | None,
    language: str,
    append: bool,
) -> int:
    """Download articles from a list of Wikipedia URLs or a URL file."""
    if urls:
        url_list = urls
    elif url_file:
        url_list = Path(url_file).read_text(encoding="utf-8").strip().splitlines()
    else:
        raise ValueError(
            "strategy='specific' requires either 'urls' (list of URLs) or "
            "'url_file' (path to a text file with one URL per line)."
        )

    url_list = [u.strip() for u in url_list if u.strip()]
    if not url_list:
        raise ValueError("No URLs provided (empty list or file).")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_ids(output) if append and output.exists() else set()

    records: list[dict[str, Any]] = []
    for url in url_list:
        parsed = _parse_wiki_url(url)
        if not parsed:
            logger.warning("Skipping unrecognised URL: %s", url)
            continue

        url_lang, title = parsed
        article = _fetch_article_text(title, language=url_lang)
        if article is None:
            continue
        if not article["text"]:
            logger.warning("Empty text for '%s' — skipping", title)
            continue

        page_id = article["page_id"]
        record_id = f"wikipedia:{url_lang}:{page_id}"
        if record_id in existing_ids:
            logger.info("Skipping duplicate: %s", record_id)
            continue

        existing_ids.add(record_id)
        records.append(
            {
                "id": record_id,
                "text": article["text"],
                "title": article["title"],
                "url": article["url"],
                "source": "wikipedia_api",
                "dataset": "wikipedia_api",
                "dataset_config": f"specific.{url_lang}",
                "language": url_lang,
                "page_id": page_id,
                "downloaded_at": _utc_now(),
                "license": DATASET_LICENSE,
                "license_url": DATASET_URL,
                "categories": [],
            }
        )

    mode = "a" if append else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    logger.info("Wrote %d specific Wikipedia articles to %s", len(records), output)
    return len(records)


def _download_degree_targeted(
    path: str | Path,
    count: int,
    language: str,
    snapshot: str,
    seed: int | None,
    max_scan: int,
    min_chars: int,
    target_degree: float,
    max_articles: int | None,
    exclude_namespaces: list[str] | None = None,
) -> int:
    """Grow a connected set of articles to meet a target average hyperlink degree.

    1. Seed with ``count // 2`` random articles from HuggingFace.
    2. Enrich to get outgoing link titles.
    3. Compute the average degree (edges within the downloaded set / article count).
    4. Iteratively add articles that are linked from the current set until the
       target degree is met or ``max_articles`` is reached.
    """
    import time as _time

    excluded = exclude_namespaces if exclude_namespaces is not None else _DEFAULT_EXCLUDE_NAMESPACES
    if excluded:
        logger.info("Excluding Wikipedia namespaces: %s", excluded)

    max_articles = max_articles or count * 5
    if target_degree <= 0:
        raise ValueError("target_degree must be positive")
    if count < 2:
        raise ValueError("Need at least 2 articles for degree calculation")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: seed with random articles
    seed_count = max(count // 2, 2)
    client = HuggingFaceWikipediaClient(
        language=language,
        snapshot=snapshot,
        seed=seed,
        max_scan=max_scan,
    )
    records = client.fetch_random(seed_count, min_chars=min_chars)

    # Phase 2: enrich to discover links
    for record in records:
        page_id = record.get("page_id", "")
        if page_id:
            titles = _fetch_outgoing_links(page_id, language=language)
            record["_linked_titles"] = titles
        _time.sleep(REQUEST_DELAY)

    # Phase 3: iterative expansion
    current_ids = {r["id"] for r in records}
    title_to_id = {r["title"]: r["id"] for r in records}

    def _compute_degree() -> float:
        """Compute average degree (edges within the set / article count)."""
        edges = 0
        for r in records:
            linked = r.get("_linked_titles", [])
            edges += sum(1 for t in linked if t in title_to_id)
        return edges / len(records) if records else 0.0

    avg_degree = _compute_degree()
    batch_size = max(5, count // 10)

    while avg_degree < target_degree and len(records) < max_articles:
        # Find candidate titles linked from current set but not yet downloaded
        candidate_scores: dict[str, int] = {}
        for r in records:
            for title in r.get("_linked_titles", []):
                if title not in title_to_id and not _is_excluded(title, excluded):
                    candidate_scores[title] = candidate_scores.get(title, 0) + 1

        if not candidate_scores:
            logger.warning(
                "No more linked articles available; stopping at degree %.2f / %d articles",
                avg_degree,
                len(records),
            )
            break

        # Pick top candidates
        top = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:batch_size]
        new_records = []
        for title in top:
            if len(records) + len(new_records) >= max_articles:
                break

            article = _fetch_article_text(title, language=language)
            if article is None or not article["text"]:
                continue

            page_id = article["page_id"]
            record_id = f"wikipedia:{language}:{page_id}"
            if record_id in current_ids:
                continue

            record = {
                "id": record_id,
                "text": article["text"],
                "title": article["title"],
                "url": article["url"],
                "source": "wikipedia_api",
                "dataset": "wikipedia_api",
                "dataset_config": f"degree.{language}",
                "language": language,
                "page_id": page_id,
                "downloaded_at": _utc_now(),
                "license": DATASET_LICENSE,
                "license_url": DATASET_URL,
                "categories": [],
            }

            # Fetch outgoing links for the new article
            links = _fetch_outgoing_links(page_id, language=language)
            record["_linked_titles"] = links
            _time.sleep(REQUEST_DELAY)

            new_records.append(record)
            current_ids.add(record_id)
            title_to_id[article["title"]] = record_id

        if not new_records:
            logger.warning("No new articles could be fetched; stopping.")
            break

        records.extend(new_records)
        avg_degree = _compute_degree()
        logger.info(
            "Degree expansion: %d articles, avg degree %.2f (target %.2f)",
            len(records),
            avg_degree,
            target_degree,
        )

    # Convert internal linked-titles to canonical "links" field
    for r in records:
        titles = r.pop("_linked_titles", None)
        if titles:
            r["links"] = [_title_to_url(t, language=language) for t in titles]

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    logger.info(
        "Wrote %d degree-targeted articles to %s (avg degree %.2f)",
        len(records),
        output,
        avg_degree,
    )
    return len(records)


# ═══════════════════════════════════════════════════════════════
# Enrich
# ═══════════════════════════════════════════════════════════════


def _fetch_outgoing_links(page_id: str, language: str = "en") -> list[str]:
    """Fetch all outgoing wiki-link page titles for a Wikipedia page."""
    import requests

    api_url = f"https://{language}.wikipedia.org/w/api.php"
    titles: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "prop": "links",
        "pageids": page_id,
        "pllimit": "max",
        "format": "json",
    }
    headers = {
        "User-Agent": "Polygraph/0.2 (https://github.com/user/polygraph; research KG construction)"
    }

    while True:
        try:
            resp = requests.get(api_url, params=params, timeout=30, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("API error for page %s: %s", page_id, exc)
            break

        pages = data.get("query", {}).get("pages", {})
        page_data = pages.get(str(page_id), pages.get(page_id, {}))
        links = page_data.get("links", [])
        for link in links:
            title = link.get("title", "")
            if title:
                titles.append(title)

        cont = data.get("continue", {})
        if "plcontinue" in cont:
            params["plcontinue"] = cont["plcontinue"]
        else:
            break

    return titles


def _title_to_url(title: str, language: str = "en") -> str:
    """Convert a Wikipedia page title to its full URL."""
    slug = quote(title.replace(" ", "_"), safe="()!,:;@&=+$-_.~'")
    return f"https://{language}.wikipedia.org/wiki/{slug}"


def enrich_wikipedia(
    input_path: str | Path,
    output_path: str | Path | None = None,
    language: str = "en",
    delay: float = REQUEST_DELAY,
    force: bool = False,
) -> int:
    """Add outgoing Wikipedia hyperlinks to a Polygraph JSONL file.

    Reads existing JSONL records (which must have a ``page_id`` field),
    queries the Wikipedia API for outgoing links, and writes enriched
    records with a ``links`` field containing full Wikipedia URLs.

    By default, skips enrichment if records already contain a ``links``
    field.  Set ``force=True`` to always re-fetch.

    Args:
        input_path: Path to existing Polygraph Wikipedia JSONL.
        output_path: Where to write enriched JSONL (defaults to overwriting ``input_path``).
        language: Wikipedia language code.
        delay: Seconds to wait between API requests (be polite to Wikipedia).
        force: If ``True``, re-enrich even if records already have links.

    Returns:
        Number of records enriched with links (0 if cached).
    """
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **kw):
            return x  # type: ignore[assignment]

    inp = Path(input_path)
    out = Path(output_path) if output_path else inp

    # Read all records
    records: list[dict[str, Any]] = []
    with inp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        logger.warning("No records found in %s", inp)
        return 0

    # Cache check: skip if records already have links (unless forced)
    if not force and _records_have_links(records):
        logger.info(
            "Skipping enrichment — %d/%d records already have links (use force=True to re-enrich)",
            sum(1 for r in records if r.get("links")),
            len(records),
        )
        return 0

    logger.info("Enriching %d records with Wikipedia outgoing links...", len(records))

    enriched_count = 0
    for record in tqdm(records, desc="Fetching links"):
        page_id = record.get("page_id")
        if not page_id:
            logger.warning("Skipping record %s — no page_id", record.get("id", "?"))
            continue

        try:
            titles = _fetch_outgoing_links(page_id, language=language)
            urls = [_title_to_url(t, language=language) for t in titles]
            record["links"] = urls
            enriched_count += 1
        except Exception as exc:
            logger.warning("Failed to fetch links for page %s: %s", page_id, exc)

        time.sleep(delay)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    logger.info(
        "Wrote %d enriched records to %s (%d with links)",
        len(records),
        out,
        enriched_count,
    )
    return enriched_count


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════


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


def _records_have_links(records: list[dict[str, Any]]) -> bool:
    """Return ``True`` if all records already have a non-empty ``links`` field."""
    if not records:
        return False
    return all(r.get("links") for r in records)


def _close_dataset(dataset: Any) -> None:
    """Shut down background workers of a streaming Hugging Face dataset."""
    for attr in ("_ex_iterable", "_iterable"):
        it = getattr(dataset, attr, None)
        if it is not None:
            for method in ("close", "cancel", "shutdown"):
                closer = getattr(it, method, None)
                if callable(closer):
                    with contextlib.suppress(Exception):
                        closer()
            with contextlib.suppress(Exception):
                setattr(dataset, attr, None)

    for method in ("cleanup_cache_files", "close", "_cleanup"):
        closer = getattr(dataset, method, None)
        if callable(closer):
            with contextlib.suppress(Exception):
                closer()

    gc.collect()


def _article_url(language: str, title: str) -> str:
    slug = quote(title.replace(" ", "_"), safe="()!,:;@&=+$-_.~'")
    return f"https://{language}.wikipedia.org/wiki/{slug}"


def _is_excluded(title: str, namespace_prefixes: list[str]) -> bool:
    """Return True if *title* starts with any excluded namespace prefix."""
    return any(title.startswith(prefix) for prefix in namespace_prefixes)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _schedule_watchdog(seconds: int = 5) -> None:
    """Force-exit the process after *seconds* if PyArrow threads keep it alive.

    HuggingFace ``datasets`` streaming spawns background threads that can
    prevent clean Python shutdown.  This watchdog gives them a grace period
    to finish, then calls ``os._exit(0)`` to terminate immediately.
    """

    def _force_exit(_signum: int, _frame: Any) -> None:
        os._exit(0)

    signal.signal(signal.SIGALRM, _force_exit)
    signal.alarm(seconds)


__all__ = [
    "HuggingFaceDownloadError",
    "HuggingFaceWikipediaClient",
    "download_wikipedia",
    "enrich_wikipedia",
]
