"""Tests for the benchmark pipeline — stage runners, downloaders, comparison.

These tests are hermetic: no network, no dependency on the real gold files.
``Data.download`` is monkeypatched to a no-op where a runner would fetch gold.
"""

from __future__ import annotations

import inspect
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from polygraph._shared import Document
from polygraph._shared.types import PreprocessResult
from polygraph.benchmark_pipeline import Benchmark, BenchmarkResult, BenchmarkRunner
from polygraph.benchmark_pipeline.chunking import runner as chunking_mod
from polygraph.benchmark_pipeline.dedup import runner as dedup_mod
from polygraph.benchmark_pipeline.quality import runner as quality_mod
from polygraph.benchmark_pipeline.rag import runner as rag_mod
from polygraph.data import Data, _benchmarks
from polygraph.pipelines import Baseline, Pipeline

ALL_STAGES = ["Dedup", "Chunking", "Resolution", "Extraction", "Quality", "RAG"]


# ── Structural guarantees ──────────────────────────────────────


def test_all_runners_construct_with_no_args() -> None:
    """tools/run_benchmarks.py constructs every runner with no arguments."""
    for attr in ALL_STAGES:
        runner = getattr(Benchmark, attr)()
        # Chunking resolves its default lazily (``None`` → standard gold path);
        # every other runner carries an explicit default dataset path.
        if attr != "Chunking":
            assert runner.dataset, f"{attr}() must have a default dataset path"


def test_all_runners_have_help() -> None:
    for attr in ALL_STAGES:
        assert getattr(Benchmark, attr).help()


def test_no_runner_is_a_stub() -> None:
    """Every stage runner's run() must be implemented."""
    modules = {
        "Dedup": dedup_mod,
        "Chunking": chunking_mod,
        "QualityFilter": quality_mod,
        "RAG": rag_mod,
    }
    for name, mod in modules.items():
        src = inspect.getsource(getattr(mod, f"{name}Runner").run)
        assert "NotImplementedError" not in src, f"{name}Runner.run is a stub"


# ── Chunking ───────────────────────────────────────────────────


def test_chunking_boundary_f1_perfect_when_no_boundaries() -> None:
    text = "EU rejects German call to boycott British lamb."
    metrics = chunking_mod._score_all(
        [{"text": text, "chunks": [{"name": "EU", "type": "ORG"}]}],
        [[text]],  # single chunk → no boundaries → perfect
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_chunking_boundary_inside_entity_is_penalized() -> None:
    text = "EU rejects German call to boycott British lamb."
    # Split "EU" in half: boundary at offset 1 lies strictly inside entity (0,2).
    metrics = chunking_mod._score_all(
        [{"text": text, "chunks": [{"name": "EU", "type": "ORG"}]}],
        [[text[:1], text[1:]]],
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


# ── Dedup scoring ──────────────────────────────────────────────


def test_dedup_score() -> None:
    y_true = ["duplicate", "duplicate", "not_duplicate"]
    y_pred = ["duplicate", "not_duplicate", "not_duplicate"]
    metrics = dedup_mod._score(y_true, y_pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(2 / 3, abs=0.001)


# ── Quality scoring ────────────────────────────────────────────


def test_quality_score() -> None:
    y_true = ["keep", "keep", "reject"]
    y_pred = ["keep", "reject", "reject"]
    metrics = quality_mod._score(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(2 / 3, abs=0.001)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


# ── Resolution runner (offline, string method) ─────────────────


def test_resolution_runner_small_gold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Data, "download", lambda *a, **k: None)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "names": ["IBM", "International Business Machines", "IBM Corp"],
                        "cluster_id": 1,
                    }
                ),
                json.dumps({"names": ["Apple", "Apple Inc"], "cluster_id": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = Benchmark.Resolution(dataset=gold).run(pipelines={"string": Baseline()})
    metrics = result["string"]
    assert metrics["method"] == "string"
    assert metrics["precision"] >= 0.0
    assert metrics["recall"] >= 0.0
    assert metrics["f1"] >= 0.0


# ── Extraction runner (offline, spaCy) ─────────────────────────


def test_extraction_runner_small_gold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Data, "download", lambda *a, **k: None)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "text": "Apple and IBM announced a merger.",
                "entities": [
                    {"name": "Apple", "type": "ORG", "start": 0, "end": 5},
                    {"name": "IBM", "type": "ORG", "start": 10, "end": 13},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = Benchmark.Extraction(dataset=gold).run(pipelines={"surface": Baseline()})
    metrics = result["surface"]
    assert "precision" in metrics and "recall" in metrics and "f1" in metrics
    assert metrics["entity_method"] == "spacy"


# ── RAG runner (fake pipeline) ─────────────────────────────────


class _FakePipeline(Pipeline):
    """Minimal pipeline whose KG contains the answer to the gold query."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_paths = []

    def preprocess(self) -> PreprocessResult:
        return PreprocessResult(
            chunks=[Document(content="Marie Curie discovered radium in Paris.")]
        )

    def build_kg(self, chunks) -> dict[str, Any]:
        return {
            "entities": [{"name": "Marie Curie"}],
            "triples": [("Marie Curie", "discovered", "radium")],
            "graph": None,
        }


def test_rag_runner_small_gold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Data, "download", lambda *a, **k: None)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "query": "Who discovered radium?",
                "answer_entity": "Marie Curie",
                "supporting_chunks": ["Marie Curie discovered radium in Paris."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = Benchmark.RAG(dataset=gold).run(
        pipelines={"fake": _FakePipeline()}, input_paths=["."], k=1
    )
    metrics = result["fake"]
    assert metrics["entity_recall_at_k"] == 1.0
    assert metrics["entity_coverage"] == 1.0
    assert metrics["chunk_recall_at_k"] == 1.0
    assert metrics["chunk_precision_at_k"] == 1.0
    assert metrics["relation_path_accuracy"] is None


# ── compare_matrix regression (config merge) ───────────────────


def test_compare_matrix_merges_chunk_and_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweeping both axes must not silently overwrite one config (bug fix)."""
    import polygraph.benchmark_pipeline.compare as compare_mod

    captured: dict[str, Any] = {}

    class _FakeResult:
        overall_score = 0.0
        num_entities = 0
        num_triples = 0
        pipeline_variant = "surface"

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self) -> _FakeResult:
            return _FakeResult()

    monkeypatch.setattr(compare_mod, "BenchmarkRunner", _FakeRunner)

    compare_mod.compare_matrix(
        variants=["surface"],
        input_paths=["data/"],
        chunkers=["sentence"],
        doc_dedup_methods=["minhash"],
    )
    preprocess = captured["extra"]["preprocess"]
    assert preprocess.chunk_method == "sentence"
    assert preprocess.doc_dedup_method == "minhash"


# ── Whole-pipeline BenchmarkRunner (fake pipeline) ─────────────


class _FakeExecPipeline(Pipeline):
    """Pipeline that writes a metrics.json, as the real pipelines do."""

    def preprocess(self):
        return []

    def build_kg(self, chunks):
        return {}

    def execute(self, input_paths=None, output_dir=None, cache=False, force=False) -> None:
        out = Path(output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            json.dumps({"overall_score": 0.5, "num_entities": 3, "num_triples": 4}),
            encoding="utf-8",
        )


def test_benchmark_runner_collects_results(tmp_path: Path) -> None:
    runner = BenchmarkRunner(
        pipeline=_FakeExecPipeline,
        input_paths=["data/"],
        output_dir=tmp_path / "out",
    )
    result: BenchmarkResult = runner.run()
    assert result.overall_score == pytest.approx(0.5)
    assert result.num_entities == 3
    assert result.num_triples == 4
    assert (tmp_path / "out" / "results_summary.json").exists()


# ── DBLP downloader pair-building (synthetic zip, no network) ──


def test_dblp_unileipzig_parser_builds_balanced_pairs(tmp_path: Path) -> None:
    zip_path = tmp_path / "dblp.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "DBLP2.csv",
            '"id","title","authors","venue","year"\n'
            '"d1","Alpha paper","Alice","DBLP",2019\n'
            '"d2","Beta paper","Bob","DBLP",2020\n',
        )
        zf.writestr(
            "ACM.csv",
            '"id","title","authors","venue","year"\n'
            '1,"Alpha paper","Alice","ACM",2019\n'
            '2,"Gamma paper","Carol","ACM",2020\n',
        )
        zf.writestr(
            "DBLP-ACM_perfectMapping.csv",
            '"idDBLP","idACM"\n"d1",1\n',
        )

    with zipfile.ZipFile(zip_path) as zf:
        records = _benchmarks._dblp_acm_records(zf)

    labels = [r["label"] for r in records]
    assert labels.count("duplicate") == 1
    assert labels.count("not_duplicate") == 1
    dup = next(r for r in records if r["label"] == "duplicate")
    assert "Alpha" in dup["text_a"] and "Alpha" in dup["text_b"]


def test_dblp_deepmatcher_parser(tmp_path: Path) -> None:
    zip_path = tmp_path / "deepmatcher.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "table.csv",
            '"id","label","title_left","authors_left","title_right","authors_right"\n'
            '0,1,"Same paper","Alice","Same paper","Alice"\n'
            '1,0,"Unique one","Bob","Unique two","Carol"\n',
        )

    with zipfile.ZipFile(zip_path) as zf:
        records = _benchmarks._dblp_acm_records(zf)

    labels = [r["label"] for r in records]
    assert labels == ["duplicate", "not_duplicate"]
