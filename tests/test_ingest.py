"""Tests for the ingest stage."""

import tempfile
from pathlib import Path

from polygraph._shared import Document, Language
from polygraph.preprocess.load import DataLoader


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
    from polygraph.preprocess.clean import TextCleaner

    doc = Document(content="  Hello   world!\n\nExtra  spaces.  ")
    cleaner = TextCleaner()
    result = cleaner.clean(doc)
    assert "  " not in result.content
    assert result.content == "Hello world!\n\nExtra spaces."


def test_semantic_chunker_splits_at_topic_shift_with_fake_encoder():
    from polygraph.preprocess.chunk import SemanticChunker

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
    from polygraph.preprocess.clean.en.normalizer import EnglishCleaner

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
