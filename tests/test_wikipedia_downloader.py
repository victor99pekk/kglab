"""Tests for the local Wikimedia sample writer without network access."""

import json

from polygraph.ingest.wikipedia import write_manifest, write_sample


def test_write_sample_and_manifest(tmp_path):
    articles = iter(
        [
            {"id": "empty", "text": "", "title": "Empty", "url": "https://example.test/empty"},
            {
                "id": "article",
                "text": "A useful article.",
                "title": "Article",
                "url": "https://example.test/article",
            },
        ]
    )
    output_path = tmp_path / "sample.jsonl"
    manifest_path = tmp_path / "sample_manifest.yaml"

    assert write_sample(articles, 1, output_path) == 1
    write_manifest(manifest_path, "en", "20231101", 1)

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["id"] == "article"
    assert record["title"] == "Article"
    manifest = manifest_path.read_text(encoding="utf-8")
    assert "language: vi" in manifest
    assert "version: 20231101-vi-1" in manifest
