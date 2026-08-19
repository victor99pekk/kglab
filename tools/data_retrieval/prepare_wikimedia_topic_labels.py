#!/usr/bin/env python3
"""Download and normalize Wikimedia's public article-topic labels.

The tool is experiment-agnostic. Callers provide a pinned dataset manifest,
taxonomy, and artifact directory. Networking uses certificate verification.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import ssl
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

USER_AGENT = "KGLab-Wikimedia-Experiment/0.1 (research dataset preparation)"


def verified_ssl_context() -> ssl.SSLContext:
    """Use uv's certifi bundle when the interpreter lacks a usable CA path."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def taxonomy_labels(taxonomy_path: Path) -> tuple[str, ...]:
    taxonomy = load_yaml(taxonomy_path)
    labels = tuple(
        str(label) for domain in taxonomy.get("domains", []) for label in domain.get("labels", [])
    )
    expected = int(taxonomy.get("label_count", 0))
    if len(labels) != expected:
        raise ValueError(f"Taxonomy declares {expected} labels but defines {len(labels)}")
    if len(labels) != len(set(labels)):
        duplicates = [label for label, count in Counter(labels).items() if count > 1]
        raise ValueError(f"Duplicate taxonomy labels: {duplicates}")
    return labels


def validate_definitions(
    manifest_path: Path,
    taxonomy_path: Path,
) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    labels = taxonomy_labels(taxonomy_path)
    files = manifest.get("files", {})
    required_files = {"taxonomy_mapping", "template_mapping", "labels_en"}
    missing = required_files - set(files)
    if missing:
        raise ValueError(f"Manifest missing files: {sorted(missing)}")
    for key, record in files.items():
        if not isinstance(record, dict):
            raise ValueError(f"Manifest file record must be a mapping: {key}")
        for field in ("id", "name", "url", "size", "md5"):
            if field not in record:
                raise ValueError(f"Manifest file {key!r} missing {field!r}")
    return {
        "taxonomy_labels": len(labels),
        "manifest_files": sorted(files),
        "source_version": manifest["source"]["version"],
        "status": "valid",
    }


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published artifact integrity, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_download(path: Path, record: dict[str, Any]) -> None:
    expected_size = int(record["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {actual_size}"
        )
    expected_md5 = str(record["md5"])
    actual_md5 = file_md5(path)
    if actual_md5 != expected_md5:
        raise ValueError(f"MD5 mismatch for {path.name}: expected {expected_md5}, got {actual_md5}")


def download_file(
    file_key: str,
    manifest_path: Path,
    raw_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    manifest = load_yaml(manifest_path)
    try:
        record = manifest["files"][file_key]
    except KeyError as exc:
        choices = ", ".join(sorted(manifest.get("files", {})))
        raise ValueError(f"Unknown file {file_key!r}; choose one of: {choices}") from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / str(record["name"])
    if destination.exists() and not overwrite:
        verify_download(destination, record)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(str(record["url"]), headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(
                request,
                timeout=60,
                context=verified_ssl_context(),
            ) as response,
            partial.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    verify_download(partial, record)
    partial.replace(destination)
    return destination


def deterministic_split(qid: str) -> str:
    """Match Wikimedia's 90/2/8 proportions using stable QID grouping."""
    bucket = int.from_bytes(hashlib.sha256(qid.encode("utf-8")).digest()[:8], "big") % 10_000
    if bucket < 9_000:
        return "train"
    if bucket < 9_200:
        return "validation"
    return "test"


def normalize_record(
    source: dict[str, Any],
    allowed_labels: set[str],
) -> dict[str, Any] | None:
    title = str(source.get("title", "")).strip()
    qid = str(source.get("qid", "")).strip()
    sitelinks = source.get("sitelinks", {})
    page_id = sitelinks.get("enwiki") if isinstance(sitelinks, dict) else None
    if not title or not qid or page_id is None:
        return None

    topics = tuple(dict.fromkeys(str(value) for value in source.get("topics", [])))
    unknown = sorted(set(topics) - allowed_labels)
    if unknown:
        raise ValueError(f"Unknown Wikimedia topics for {qid}: {unknown}")
    if not topics:
        return None

    return {
        "article_id": f"wikipedia:en:{page_id}",
        "page_id": int(page_id),
        "title": title,
        "qid": qid,
        "article_revision_id": int(source["article_revid"]),
        "talk_page_id": int(source["talk_pid"]),
        "talk_revision_id": int(source["talk_revid"]),
        "topic_ids": list(topics),
        "wikiprojects": list(dict.fromkeys(str(v) for v in source.get("wp_templates", []))),
        "split": deterministic_split(qid),
        "confidence": 1.0,
        "provenance": {
            "method": "wikimedia_wikiproject_mapping",
            "figshare_article_id": 10248344,
            "figshare_version": 4,
            "source_snapshot": "2020-05-24",
            "license": "CC0",
        },
    }


def prepare(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    taxonomy_path: Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run the tool's `download labels_en` command first."
        )
    if (output_path.exists() or summary_path.exists()) and not overwrite:
        raise FileExistsError("Prepared output exists; pass --overwrite to replace it")

    allowed_labels = set(taxonomy_labels(taxonomy_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    skipped = 0
    processed = 0

    try:
        with bz2.open(input_path, mode="rt", encoding="utf-8") as source_handle, \
            temporary.open("w", encoding="utf-8") as output_handle:
            for line_number, line in enumerate(source_handle, start=1):
                    if not line.strip():
                        continue
                    source = json.loads(line)
                    if not isinstance(source, dict):
                        raise ValueError(f"Expected JSON object at source line {line_number}")
                    record = normalize_record(source, allowed_labels)
                    if record is None:
                        skipped += 1
                        continue
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1
                    counts[record["split"]] += 1
                    label_counts.update(record["topic_ids"])
                    if limit is not None and processed >= limit:
                        break
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    missing_labels = sorted(allowed_labels - set(label_counts))
    summary = {
        "records": processed,
        "skipped": skipped,
        "splits": dict(sorted(counts.items())),
        "label_count": len(allowed_labels),
        "labels_observed": len(label_counts),
        "labels_missing": missing_labels,
        "label_frequencies": dict(sorted(label_counts.items())),
        "source": {
            "figshare_article_id": 10248344,
            "figshare_version": 4,
            "input": str(input_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Pinned dataset manifest YAML",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
        help="Frozen topic taxonomy YAML",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing raw/ and prepared/ outputs",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Validate pinned manifest and 64-label taxonomy")

    download = commands.add_parser("download", help="Download and verify one pinned source file")
    download.add_argument(
        "file",
        choices=("taxonomy_mapping", "template_mapping", "labels_en"),
    )
    download.add_argument("--overwrite", action="store_true")

    prepare_command = commands.add_parser("prepare", help="Normalize labels and assign QID splits")
    prepare_command.add_argument("--input", type=Path)
    prepare_command.add_argument("--output", type=Path)
    prepare_command.add_argument("--summary", type=Path)
    prepare_command.add_argument("--limit", type=int)
    prepare_command.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = args.manifest.resolve()
    taxonomy = args.taxonomy.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    raw_dir = artifacts_dir / "raw"
    prepared_dir = artifacts_dir / "prepared"
    try:
        if args.command == "validate":
            result = validate_definitions(manifest, taxonomy)
        elif args.command == "download":
            result = {
                "downloaded": str(
                    download_file(
                        args.file,
                        manifest,
                        raw_dir,
                        overwrite=args.overwrite,
                    )
                )
            }
        else:
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            input_path = args.input or (raw_dir / "labeled_enwiki_with_topics_metadata.json.bz2")
            output_path = args.output or (prepared_dir / "article_topic_labels.jsonl")
            summary_path = args.summary or (prepared_dir / "summary.json")
            result = prepare(
                input_path,
                output_path,
                summary_path,
                taxonomy,
                limit=args.limit,
                overwrite=args.overwrite,
            )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
