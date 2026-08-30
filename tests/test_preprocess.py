"""Tests for the method-override-first preprocessing API on pipelines."""

from __future__ import annotations

import json
from pathlib import Path

from kglab._shared import Document
from kglab.pipelines import Baseline, Pipeline
from kglab.preprocess import chunk, clean, load


class _SmallChunks(Baseline):
    """Baseline variant that overrides ``preprocess()`` directly."""

    def preprocess(self) -> list[Document]:
        docs = clean.normalize(load.from_paths(self.input_paths))
        return chunk.by_sentence(docs, target_tokens=200, overlap_tokens=60)


def _make_corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "id": "doc0",
                    "text": (
                        "Alice founded Acme Corporation in 1999 and served as its "
                        "first chief executive officer through years of steady growth. "
                        "Bob joined the company in 2010 as a senior engineer and later "
                        "became the head of research in the London office."
                    ),
                }
            )
            + "\n"
        )
    return path


def test_baseline_exposes_preprocess_summary():
    summary = Baseline().preprocess_summary
    assert "chunk_method" not in summary
    assert summary["chunk_target_tokens"] == 450
    assert summary["quality_min_chars"] == 200


def test_subclass_can_override_preprocess_summary():
    """Subclasses that change the architecture can override the summary."""

    class Sentence(Baseline):
        def preprocess(self) -> list[Document]:
            return []

        @property
        def preprocess_summary(self) -> dict:
            return {"chunk_method": "sentence", "chunk_target_tokens": 200}

    summary = Sentence().preprocess_summary
    assert summary["chunk_method"] == "sentence"
    assert summary["chunk_target_tokens"] == 200


def test_custom_pipeline_has_no_preprocess_summary():
    class Bare(Pipeline):
        def preprocess(self) -> list[Document]:
            return []

        def build_kg(self, chunks):
            return {"graph": None, "entities": [], "triples": []}

    assert getattr(Bare(), "preprocess_summary", None) is None


def test_subclass_preprocess_override_runs(tmp_path):
    pipe = _SmallChunks(input_paths=[str(_make_corpus(tmp_path))])
    chunks = pipe.preprocess()
    assert chunks
    assert all(isinstance(c, Document) for c in chunks)
    assert all(c.content.strip() for c in chunks)
