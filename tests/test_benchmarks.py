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

from kglab._shared import Document
from kglab._shared.types import PreprocessResult
from kglab.benchmark_pipeline import Benchmark, BenchmarkResult, BenchmarkRunner
from kglab.benchmark_pipeline.chunking import runner as chunking_mod
from kglab.benchmark_pipeline.dedup import runner as dedup_mod
from kglab.benchmark_pipeline.quality import runner as quality_mod
from kglab.benchmark_pipeline.rag import runner as rag_mod
from kglab.data import Data, _benchmarks
from kglab.pipelines import Baseline, Pipeline

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


def test_pipeline_registry_has_baseline_alias() -> None:
    """'baseline' is the default variant name — it must resolve to Baseline.

    ``ExperimentConfig`` defaults to ``pipeline_variant="baseline"`` and the
    example YAML configs use ``variant: baseline``, so the registry must know
    that name (a ``BenchmarkRunner.from_config`` with it used to crash).
    """
    from kglab.pipelines import PIPELINE_REGISTRY, Baseline

    assert PIPELINE_REGISTRY["baseline"] is Baseline
    assert BenchmarkRunner._resolve_pipeline("baseline") is Baseline


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


def test_compare_matrix_sweeps_resolutions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """compare_matrix sweeps resolution methods across the variant."""
    import kglab.benchmark_pipeline.compare as compare_mod

    captured: list[dict[str, Any]] = []

    class _FakeResult:
        overall_score = 0.0
        num_entities = 0
        num_triples = 0
        pipeline_variant = "surface"

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

        def run(self) -> _FakeResult:
            return _FakeResult()

    monkeypatch.setattr(compare_mod, "BenchmarkRunner", _FakeRunner)

    compare_mod.compare_matrix(
        variants=["surface"],
        input_paths=["data/"],
        output_dir=str(tmp_path),
        resolutions=["string", "embedding"],
    )
    methods = sorted(k["resolution"].method for k in captured)
    assert methods == ["embedding", "string"]


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


# ── Model reuse (semantic paths must load the model once) ──────


def test_chunking_semantic_builds_one_chunker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semantic chunking must reuse one model-backed chunker per run."""
    from kglab.preprocess.chunk import SemanticChunker as RealChunker

    init_count = {"n": 0}

    class SpyChunker(RealChunker):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            init_count["n"] += 1
            kwargs.setdefault("encoder", lambda texts: [[1.0, 0.0]] * len(texts))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(chunking_mod, "SemanticChunker", SpyChunker)
    gold = [
        {
            "text": "One sentence about a topic. Two sentences about the same topic.",
            "chunks": [],
        }
        for _ in range(4)
    ]
    out = chunking_mod._chunk_records(gold, "semantic", {})
    assert init_count["n"] == 1, "SemanticChunker must be built once per run"
    assert len(out) == 4


def test_dedup_semantic_encoder_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The embedding model must load once and be shared across pairs."""
    loads = {"n": 0}

    def fake_encoder(texts):
        return [[1.0, 0.0]] * len(texts)

    def fake_load(model_name: str):
        loads["n"] += 1
        return fake_encoder

    monkeypatch.setattr(dedup_mod, "_load_sentence_encoder", fake_load)
    dedup_mod._MODEL_CACHE.clear()
    try:
        first = dedup_mod._semantic_encoder("some-model")
        second = dedup_mod._semantic_encoder("some-model")
        assert first is second
        assert loads["n"] == 1
    finally:
        dedup_mod._MODEL_CACHE.clear()


def test_runner_errors_propagate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A broken pipeline must raise — never silently return zeroed metrics."""
    monkeypatch.setattr(Data, "download", lambda *a, **k: None)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"text": "hello world foo bar baz.", "chunks": []}) + "\n",
        encoding="utf-8",
    )

    class Bad(Baseline):
        @property
        def preprocess_summary(self):
            return {"chunk_method": "does_not_exist"}

    with pytest.raises(ValueError, match="Unknown chunk method"):
        Benchmark.Chunking(dataset=gold).run(pipelines={"bad": Bad()})


def test_dedup_predict_pair_semantic_uses_shared_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic/layered pairs use the cached encoder (no per-pair model load)."""

    def encode(texts: list[str]) -> list[list[float]]:
        vectors = {
            "hello world foo bar": [1.0, 0.0],
            "completely different text": [0.0, 1.0],
            "unrelated other text": [0.5, 0.5],
        }
        return [vectors.get(t, [0.0, 0.0]) for t in texts]

    monkeypatch.setattr(dedup_mod, "_semantic_encoder", lambda name: encode)
    assert (
        dedup_mod._predict_pair("hello world foo bar", "hello world foo bar", "semantic", 0.85)
        == "duplicate"
    )
    assert (
        dedup_mod._predict_pair(
            "completely different text", "unrelated other text", "layered", 0.85
        )
        == "not_duplicate"
    )


# ── Extraction benchmark family ───────────────────────────────


def test_extraction_golds_registered() -> None:
    """The extraction benchmark exposes a pick-your-gold family (test splits)."""
    from kglab.data import DATASET_REGISTRY

    for key in ("bench_ner_test", "bench_ner_wikiann", "bench_ner_fewnerd"):
        assert key in DATASET_REGISTRY

    text = Benchmark.Extraction.golds()
    assert "bench_ner_test" in text
    assert "bench_ner_wikiann" in text
    assert "bench_ner_fewnerd" in text
    assert "TEST split" in text


def test_span_metrics_is_schema_scoped_with_aliases() -> None:
    """Labels outside the gold's set are not scored; aliases normalize names."""
    from kglab.benchmark_pipeline.extraction import runner as extraction_mod

    allowed = {"PER", "ORG", "MISC"}
    gold = [(0, 3, "ORG", "IBM"), (10, 13, "PER", "Amy")]
    predicted = [
        (0, 3, "ORG", "IBM"),  # TP, type OK
        (10, 13, "PERSON", "Amy"),  # TP (PERSON -> PER via alias), type OK
        (5, 8, "PERSON", "Bob"),  # FP (wrong span)
        (0, 3, "CONCEPT", "stuff"),  # outside schema -> dropped
        (20, 25, "DATE", "today"),  # outside schema -> dropped
    ]
    m = extraction_mod._span_metrics(gold, predicted, allowed, extraction_mod.NER_LABEL_ALIASES)

    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == 1.0
    assert m["type_accuracy"] == 1.0
    assert m["n_predicted"] == 5
    assert m["n_scored"] == 3
    assert m["n_dropped"] == 2


def test_span_metrics_aggregates_per_record() -> None:
    """The same span in different records counts as separate true positives."""
    from kglab.benchmark_pipeline.extraction import runner as extraction_mod

    gold = [(0, 2, "ORG", "EU")]
    predicted = [(0, 2, "ORG", "EU")]
    counts = [
        extraction_mod._count_record(gold, predicted, {"ORG"}, None),
        extraction_mod._count_record(gold, predicted, {"ORG"}, None),
        extraction_mod._count_record(gold, [], {"ORG"}, None),
    ]
    m = extraction_mod._aggregate_metrics(counts)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["precision"] == 1.0
    assert m["n_scored"] == 2


# ── Result rendering ───────────────────────────────────────────


def test_render_stage_table_lists_pipelines_and_marks_best() -> None:
    """Tables must be aligned, contain metric headers, and mark the best."""
    from kglab.benchmark_pipeline.render import format_stage

    text = format_stage(
        "chunking",
        {
            "sentence": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "runtime_seconds": 1.0},
            "semantic": {"precision": 0.9, "recall": 0.7, "f1": 0.79, "runtime_seconds": 2.0},
        },
    )
    assert "pipeline" in text and "precision" in text and "recall" in text and "f1" in text
    assert "sentence" in text and "semantic" in text
    # Best headline (f1) is marked with a star; the lower one is not.
    assert "0.790 *" in text
    assert "0.500 " in text or "0.500" in text
    assert "* = best f1" in text
    # Column alignment must stay intact (numeric column right-aligned).
    lines = [line for line in text.splitlines() if "sentence" in line or "semantic" in line]
    assert all(" | " in line for line in lines)


def test_render_all_six_stages_have_metadata() -> None:
    from kglab.benchmark_pipeline.render import STAGE_META, format_stages

    assert set(STAGE_META) == {
        "dedup",
        "chunking",
        "extraction",
        "resolution",
        "quality",
        "rag",
    }
    text = format_stages(
        {
            "dedup": {
                "minhash": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "runtime_seconds": 0.1}
            }
        }
    )
    assert "Dedup" in text and "minhash" in text


def test_render_empty_results_does_not_crash() -> None:
    from kglab.benchmark_pipeline.render import format_stage

    assert "no results" in format_stage("quality", {})


def test_chunking_max_records_scores_a_slice_of_gold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """max_records caps how many bundled gold records a stage scores."""
    monkeypatch.setattr(Data, "download", lambda *a, **k: None)
    # Pin the chunk method so the test stays hermetic (no model download).
    monkeypatch.setattr(chunking_mod, "_chunk_config", lambda pipeline: ("sentence", {}))
    gold = tmp_path / "chunking_gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps({"text": f"sentence number {i} with entity Alpha.", "chunks": []})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = Benchmark.Chunking(dataset=gold).run(
        pipelines={"p": Baseline()},
        max_records=3,
    )["p"]
    assert metrics["n_samples"] == 3


def test_extraction_max_records_scores_a_slice_of_gold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """max_records caps the gold records passed to the extraction scorer."""
    from kglab.benchmark_pipeline.extraction import runner as extraction_mod

    monkeypatch.setattr(Data, "download", lambda *a, **k: None)
    gold = tmp_path / "ner_gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps({"text": f"sentence number {i} with entity Alpha.", "entities": []})
            for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    seen: dict[str, int] = {}

    def spy(
        pipeline: Pipeline, records: list[dict[str, Any]], allowed_types: set[str]
    ) -> dict[str, Any]:
        seen["n"] = len(records)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "runtime_seconds": 0.0}

    monkeypatch.setattr(extraction_mod, "_benchmark_pipeline", spy)
    Benchmark.Extraction(dataset=gold).run(pipelines={"p": Baseline()}, max_records=4)
    assert seen["n"] == 4


# ── StageResult ────────────────────────────────────────────────


def test_stage_result_str_renders_the_report() -> None:
    """print(result) shows the aligned table with the best metric marked."""
    from kglab.benchmark_pipeline import StageResult

    result = StageResult(
        stage="dedup",
        results={
            "minhash": {
                "precision": 1.0,
                "recall": 0.229,
                "f1": 0.373,
                "runtime_seconds": 3.1,
            }
        },
    )
    text = str(result)
    assert "Dedup" in text and "precision" in text and "minhash" in text
    assert "0.373 *" in text
    # Padded with blank lines so printed results don't run together.
    assert text.startswith("\n") and text.endswith("\n")


def test_stage_result_accessors() -> None:
    """__getitem__, headline, best_pipeline, to_dict and pipelines work."""
    from kglab.benchmark_pipeline import StageResult

    result = StageResult(
        stage="chunking",
        results={
            "a": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "runtime_seconds": 1.0},
            "b": {"precision": 0.9, "recall": 0.7, "f1": 0.79, "runtime_seconds": 2.0},
        },
    )
    assert result.pipelines == ["a", "b"]
    assert result["b"]["f1"] == 0.79
    assert result.headline("b") == 0.79
    assert result.best_pipeline() == "b"
    assert result.to_dict() == result.results
    assert "chunking" in repr(result) and "a" in repr(result)


def test_stage_result_is_thin_not_a_dict() -> None:
    """Deliberately dict-like only via __getitem__ — no Mapping emulation."""
    from kglab.benchmark_pipeline import StageResult

    result = StageResult(stage="quality", results={"x": {"accuracy": 1.0}})
    assert not hasattr(result, "values")
    assert not hasattr(result, "items")
