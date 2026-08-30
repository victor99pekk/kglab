"""Tests for the Wikipedia sampler API (``kglab.data``)."""

import pytest

import kglab.data._wikipedia as wiki
from kglab.data import (
    Data,
    DegreeSampler,
    RandomSampler,
    SpecificSampler,
    _api,
    sampler_from_config,
)
from kglab.data._wikipedia import download_wikipedia

# ── Sampler config validation ─────────────────────────────────


def test_random_sampler_validates_fields():
    with pytest.raises(ValueError, match="count must be positive"):
        RandomSampler(count=0)
    with pytest.raises(ValueError, match="max_scan"):
        RandomSampler(max_scan=0)


def test_degree_sampler_validates_fields():
    with pytest.raises(ValueError, match="at least 2"):
        DegreeSampler(count=1)
    with pytest.raises(ValueError, match="target_degree"):
        DegreeSampler(target_degree=0)
    with pytest.raises(ValueError, match="max_articles"):
        DegreeSampler(count=10, max_articles=5)


def test_specific_sampler_defaults_to_no_urls():
    assert SpecificSampler().urls is None
    assert SpecificSampler(url_file="urls.txt").url_file == "urls.txt"


# ── sampler_from_config ───────────────────────────────────────


def test_sampler_from_config_accepts_name_string():
    assert isinstance(sampler_from_config("random"), RandomSampler)
    assert isinstance(sampler_from_config("degree"), DegreeSampler)
    assert isinstance(sampler_from_config("specific"), SpecificSampler)


def test_sampler_from_config_accepts_dict():
    sampler = sampler_from_config({"type": "degree", "count": 10, "target_degree": 5.0})
    assert isinstance(sampler, DegreeSampler)
    assert sampler.count == 10
    assert sampler.target_degree == 5.0


def test_sampler_from_config_passes_instances_through():
    original = RandomSampler(count=3)
    assert sampler_from_config(original) is original


def test_sampler_from_config_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown sampler 'cluster'"):
        sampler_from_config({"type": "cluster", "count": 5})


# ── download_wikipedia dispatch ───────────────────────────────


def test_download_wikipedia_rejects_invalid_sampler_type(tmp_path):
    with pytest.raises(TypeError, match="RandomSampler, SpecificSampler, or DegreeSampler"):
        download_wikipedia(tmp_path / "out.jsonl", sampler="random")  # type: ignore[arg-type]


def test_download_wikipedia_rejects_old_flat_kwargs(tmp_path):
    # Old API (count=, strategy=, ...) must fail loudly, not be silently ignored.
    with pytest.raises(TypeError, match="count"):
        download_wikipedia(tmp_path / "out.jsonl", count=5)
    with pytest.raises(TypeError, match="strategy"):
        download_wikipedia(tmp_path / "out.jsonl", strategy="degree")


def test_download_wikipedia_specific_requires_urls(tmp_path):
    with pytest.raises(ValueError, match="requires either 'urls'"):
        download_wikipedia(tmp_path / "out.jsonl", sampler=SpecificSampler())


def test_download_wikipedia_defaults_to_random_sampler(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_random(path, sampler, language, snapshot, append):
        seen.update(sampler=sampler, language=language, snapshot=snapshot, append=append)
        return 0

    monkeypatch.setattr(wiki, "_download_random", fake_random)
    download_wikipedia(tmp_path / "out.jsonl", language="vi")
    assert isinstance(seen["sampler"], RandomSampler)
    assert seen["language"] == "vi"
    assert seen["snapshot"] == wiki.DEFAULT_SNAPSHOT


def test_download_wikipedia_dispatches_specific_sampler(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_specific(path, sampler, append):
        seen.update(sampler=sampler, append=append)
        return 0

    monkeypatch.setattr(wiki, "_download_specific", fake_specific)
    download_wikipedia(
        tmp_path / "out.jsonl", sampler=SpecificSampler(urls=["https://en.wikipedia.org/wiki/A"])
    )
    assert isinstance(seen["sampler"], SpecificSampler)


# ── Data.download forwarding ──────────────────────────────────


def test_data_download_forwards_sampler(monkeypatch, tmp_path):
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
    result = Data.download(
        "wikipedia", path=str(tmp_path / "out.jsonl"), sampler=RandomSampler(count=3)
    )
    assert seen["sampler"] == RandomSampler(count=3)
    assert result["downloaded"] == 1
    # Enrichment always runs after download — no enrich=True needed.
    assert len(enrich_calls) == 1
    assert enrich_calls[0]["input_path"] == tmp_path / "out.jsonl"
