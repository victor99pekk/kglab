#!/usr/bin/env python3
"""Retrieve revision-matched Wikipedia text and body links for a KG1 pilot."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_LABELS = EXPERIMENT_DIR / "artifacts" / "prepared" / "article_topic_labels.jsonl"
DEFAULT_TAXONOMY = EXPERIMENT_DIR / "wikimedia_topics_64.yaml"
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR / "artifacts" / "kg1_revision_pilot" / "retrieved_articles.jsonl"
)
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "vietnam-ai-hackathon-kg1/0.1 "
    "(revision-matched Wikimedia topic classification experiment)"
)
IGNORED_CLASSES = {
    "ambox",
    "catlinks",
    "hatnote",
    "infobox",
    "metadata",
    "mw-editsection",
    "navbox",
    "reflist",
    "sidebar",
    "thumb",
    "toc",
    "vertical-navbox",
}


def taxonomy_labels(path: Path) -> list[str]:
    taxonomy = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        label
        for domain in taxonomy["domains"]
        for label in domain["labels"]
    ]


def select_records(
    labels_path: Path,
    topics: list[str],
    *,
    per_topic: int,
    max_articles: int,
    split: str,
) -> list[dict[str, Any]]:
    """Greedily cover underrepresented topics while streaming the large label file."""
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    allowed = set(topics)

    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] != split:
                continue
            record_topics = [
                topic
                for topic in record["topic_ids"]
                if topic in allowed and counts[topic] < per_topic
            ]
            if not record_topics:
                continue

            selected.append(record)
            counts.update(record_topics)
            if len(selected) >= max_articles:
                break
            if all(counts[topic] >= per_topic for topic in topics):
                break

    if not selected:
        raise ValueError(f"No {split!r} records selected from {labels_path}")
    return selected


def request_json(
    session: requests.Session,
    params: dict[str, Any],
    *,
    retries: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"MediaWiki request failed after {retries} attempts") from last_error


def parse_target_title(anchor: Any) -> str:
    title = str(anchor.get("title", "")).strip()
    if title:
        return title.split("#", 1)[0]

    href = str(anchor.get("href", ""))
    parsed = urlparse(href)
    if not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ").split("#", 1)[0]


def extract_text_and_links(parsed_html: str) -> tuple[str, list[dict[str, str]]]:
    """Extract article prose and anchor surfaces from revision-rendered HTML."""
    soup = BeautifulSoup(parsed_html, "lxml")
    for tag in soup.select("script, style, sup, table, figure"):
        tag.decompose()
    for class_name in IGNORED_CLASSES:
        for tag in soup.select(f".{class_name}"):
            tag.decompose()

    blocks = soup.select("p, h2, h3")
    text_parts: list[str] = []
    links: list[dict[str, str]] = []
    seen_links: set[tuple[str, str]] = set()

    for block in blocks:
        block_text = block.get_text(" ", strip=True)
        if block_text:
            text_parts.append(block_text)
        for anchor in block.select("a[href]"):
            if "new" in anchor.get("class", []):
                continue
            surface = anchor.get_text(" ", strip=True)
            title = parse_target_title(anchor)
            if not surface or not title or ":" in title:
                continue
            key = (surface, title)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append({"surface": surface, "title": title})

    return "\n".join(text_parts), links


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def resolve_qids(
    session: requests.Session,
    titles: list[str],
) -> dict[str, tuple[str, str]]:
    """Return requested title -> (resolved canonical title, QID)."""
    resolved: dict[str, tuple[str, str]] = {}
    for batch in chunks(list(dict.fromkeys(titles)), 50):
        payload = request_json(
            session,
            {
                "action": "query",
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "redirects": 1,
                "titles": "|".join(batch),
                "format": "json",
                "formatversion": 2,
            },
        )
        query = payload["query"]
        aliases = {title: title for title in batch}
        for normalized in query.get("normalized", []):
            aliases[normalized["from"]] = normalized["to"]
        for redirect in query.get("redirects", []):
            for requested, current in list(aliases.items()):
                if current == redirect["from"]:
                    aliases[requested] = redirect["to"]

        page_map = {
            page["title"]: (
                page["title"],
                page.get("pageprops", {}).get("wikibase_item", ""),
            )
            for page in query["pages"]
            if not page.get("missing")
        }
        for requested, canonical in aliases.items():
            if canonical in page_map:
                resolved[requested] = page_map[canonical]
    return resolved


def retrieve_record(
    session: requests.Session,
    label_record: dict[str, Any],
) -> dict[str, Any]:
    revision_id = int(label_record["article_revision_id"])
    payload = request_json(
        session,
        {
            "action": "parse",
            "oldid": revision_id,
            "prop": "text|revid|displaytitle",
            "disableeditsection": 1,
            "disabletoc": 1,
            "format": "json",
            "formatversion": 2,
        },
    )
    parsed = payload["parse"]
    if int(parsed["revid"]) != revision_id:
        raise ValueError(
            f"Requested revision {revision_id}, received {parsed['revid']}"
        )

    text, raw_links = extract_text_and_links(parsed["text"])
    qids = resolve_qids(session, [link["title"] for link in raw_links])
    links = [
        {
            "surface": link["surface"],
            "title": qids[link["title"]][0],
            "qid": qids[link["title"]][1],
        }
        for link in raw_links
        if link["title"] in qids and qids[link["title"]][1]
    ]
    return {
        "article_id": label_record["article_id"],
        "page_id": label_record["page_id"],
        "title": label_record["title"].replace("_", " "),
        "qid": label_record["qid"],
        "article_revision_id": revision_id,
        "split": label_record["split"],
        "topic_ids": label_record["topic_ids"],
        "text": text,
        "links": links,
        "retrieval_provenance": {
            "api": API_URL,
            "method": "action_parse_oldid",
            "revision_matched": True,
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--per-topic", type=int, default=1)
    parser.add_argument("--max-articles", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = taxonomy_labels(args.taxonomy)
    selected = select_records(
        args.labels,
        topics,
        per_topic=args.per_topic,
        max_articles=args.max_articles,
        split=args.split,
    )

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retrieved: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        article = retrieve_record(session, record)
        if not article["text"]:
            raise ValueError(f"Empty article text for {record['article_id']}")
        retrieved.append(article)
        print(
            f"[{index}/{len(selected)}] {article['title']}: "
            f"{len(article['text'])} chars, {len(article['links'])} QID links"
        )

    write_jsonl(args.output, retrieved)
    covered = sorted({topic for record in retrieved for topic in record["topic_ids"]})
    print(f"Retrieved {len(retrieved)} revision-matched articles")
    print(f"Covered {len(covered)} / {len(topics)} topics")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

