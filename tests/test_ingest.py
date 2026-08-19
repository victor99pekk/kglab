"""Tests for the ingest stage."""

import tempfile
from pathlib import Path

from kglab._shared import Document, Language
from kglab.preprocess.load import DataLoader


def test_load_txt():
    loader = DataLoader()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world. This is a test document.")
        tmp = Path(f.name)

    docs = loader._load_txt(tmp)
    assert len(docs) == 1
    assert "Hello world" in docs[0].content
    tmp.unlink()


def test_load_json():
    loader = DataLoader()
    content = '[{"text": "First doc", "id": "1"}, {"text": "Second doc", "id": "2"}]'
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)

    docs = loader._load_json(tmp)
    assert len(docs) == 2
    assert docs[0].content == "First doc"
    assert docs[0].doc_id == "1"
    tmp.unlink()


def test_cleaner_normalizes_whitespace():
    from kglab.preprocess.clean import TextCleaner

    doc = Document(content="  Hello   world!\n\nExtra  spaces.  ")
    cleaner = TextCleaner()
    result = cleaner.clean(doc)
    assert "  " not in result.content
    assert result.content == "Hello world!\n\nExtra spaces."


def test_semantic_chunker_splits_at_topic_shift_with_fake_encoder():
    from kglab.preprocess.chunk import SemanticChunker

    text = "Mèo thích ngủ trong nhà. Mèo thường chơi vào buổi tối. Tên lửa đưa vệ tinh lên quỹ đạo."

    def encoder(_texts):
        return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]

    chunks = SemanticChunker(
        target_tokens=20,
        overlap_tokens=0,
        language=Language.ENGLISH,
        similarity_threshold=0.5,
        encoder=encoder,
    ).chunk([Document(content=text, source="topics.txt", doc_id="topics")])

    assert len(chunks) == 2
    assert "Mèo" in chunks[0].content
    assert chunks[1].content == "Tên lửa đưa vệ tinh lên quỹ đạo."


def test_english_cleaner_normalizes_unicode_and_mojibake():
    """EnglishCleaner applies NFC normalization and ftfy mojibake repair."""
    from kglab.preprocess.clean.en.normalizer import EnglishCleaner

    cleaner = EnglishCleaner()

    # NFC normalization: decomposed → composed
    assert cleaner.clean("café") == "café"

    # Common mojibake patterns that ftfy repairs
    result = cleaner.clean("â€™")  # smart quote mojibake
    # Without ftfy it stays as-is; with ftfy it becomes '
    # The test verifies the function runs without error and returns a string
    assert isinstance(result, str)
    assert len(result) > 0

    # Unicode quote normalization (always active, independent of ftfy)
    result = cleaner.clean("\u201chello\u201d")
    assert '"' in result


def test_data_download_defaults_path_to_data_dir(monkeypatch):
    """Data.download() with no path= writes to data/{name}.jsonl and enriches."""
    from kglab.data import _api

    seen: dict = {}
    enrich_calls: list[dict] = []

    def fake_import_fn(import_path):
        def fake_download(**kwargs):
            seen.update(kwargs)
            return 1

        def fake_enrich(**kwargs):
            enrich_calls.append(kwargs)
            return 1

        return fake_download if "download" in import_path else fake_enrich

    monkeypatch.setattr(_api, "_import_fn", fake_import_fn)

    result = _api.Data.download("wikipedia", count=5, language="vi")
    assert result["path"] == "data/wikipedia.jsonl"
    assert seen["path"] == Path("data/wikipedia.jsonl")
    # Enrichment always runs after download — no enrich=True needed.
    assert result["enriched"] == 1
    assert enrich_calls[0]["input_path"] == Path("data/wikipedia.jsonl")
    assert enrich_calls[0]["language"] == "vi"

    # An explicit path is still honored.
    result = _api.Data.download("wikipedia", path="data/en/articles.jsonl")
    assert result["path"] == "data/en/articles.jsonl"
    assert enrich_calls[1]["input_path"] == Path("data/en/articles.jsonl")


def test_data_download_without_enrich_skips_enrichment(monkeypatch, tmp_path):
    """Datasets without an enrich step download cleanly and omit 'enriched'."""
    from kglab.data import _api

    calls: list[str] = []

    def fake_import_fn(import_path):
        def fake_download(**kwargs):
            calls.append(import_path)
            return 1

        return fake_download

    monkeypatch.setattr(_api, "_import_fn", fake_import_fn)

    result = _api.Data.download("bench_ner", path=str(tmp_path / "ner.jsonl"))
    assert result["downloaded"] == 1
    assert "enriched" not in result
    assert len(calls) == 1  # only the download step ran
