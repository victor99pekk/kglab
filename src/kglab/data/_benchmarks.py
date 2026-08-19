"""Benchmark gold datasets — download and cache for pipeline benchmarking."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

CONLL_TRAIN_URL = "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train"
CONLL_TEST_URL = "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb"
# DBLP-ACM has no stable single host — try these mirrors in order.
DBLP_ACM_MIRRORS = [
    "https://dbs.uni-leipzig.de/file/DBLP-ACM.zip",
    "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/datasets/DBLP-ACM.zip",
]
HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"


def _skip_if_cached(path: Path, force: bool) -> bool:
    if not force and path.exists() and path.stat().st_size > 0:
        logger.info("Already cached — skipping download: %s", path)
        return True
    return False


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


# ═══════════════════════════════════════════════════════════════
# NER — CoNLL-2003
# ═══════════════════════════════════════════════════════════════


def download_conll_ner(path: str | Path, force: bool = False) -> int:
    """Download the CoNLL-2003 TRAIN split as NER gold (``bench_ner``)."""
    return _download_conll(CONLL_TRAIN_URL, Path(path), force)


def download_conll_ner_test(path: str | Path, force: bool = False) -> int:
    """Download the CoNLL-2003 TEST split as NER gold (``bench_ner_test``).

    The official held-out test split (``eng.testb``) is what published NER
    numbers (spaCy ~0.85, BERT-NER ~0.91) are reported on. Same format and
    parser as the train gold, so it is scored identically.
    """
    return _download_conll(CONLL_TEST_URL, Path(path), force)


def _download_conll(url: str, target: Path, force: bool) -> int:
    """Download a CoNLL-2003 split and write it as entity-span JSONL gold."""
    if _skip_if_cached(target, force):
        return 0

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        urlretrieve(url, tmp.name)
    records = _parse_conll(tmp.name)
    os.unlink(tmp.name)
    return _write_jsonl(target, records)


def _parse_conll(path: str) -> list[dict[str, Any]]:
    """Parse a CoNLL-2003 BIO-tagged file into entity-span records."""
    records: list[dict[str, Any]] = []
    tokens: list[str] = []
    labels: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    records.append(_bio_to_entities(" ".join(tokens), tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) >= 4:
                tokens.append(parts[0])
                labels.append(parts[3])
        if tokens:
            records.append(_bio_to_entities(" ".join(tokens), tokens, labels))
    return records


def _bio_records_from_hf(ds: Any, tag_field: str) -> list[dict[str, Any]]:
    """Convert a HuggingFace token-level NER dataset into entity-span records.

    ``ds`` must expose ``tokens`` and an integer ``ner_tags``-style field whose
    ``feature.names`` maps ids to IOB2 labels.
    """
    names = ds.features[tag_field].feature.names
    records: list[dict[str, Any]] = []
    for example in ds:
        labels = [names[i] for i in example[tag_field]]
        records.append(_bio_to_entities(" ".join(example["tokens"]), example["tokens"], labels))
    return records


def download_wikiann_ner(path: str | Path, force: bool = False) -> int:
    """Download the wikiann (English, TEST split) NER gold (``bench_ner_wikiann``).

    Free, Wikipedia-derived NER with three broad types (PER/ORG/LOC) — a good
    held-out benchmark for broad entity extractors (spaCy's PERSON/GPE/LOC/ORG
    map onto it via the benchmark's label aliases).
    """
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    # ``wikiann`` is a no-namespace repo; datasets>=5 cannot parse its URI, so
    # fetch the English test parquet directly and load it as a local dataset.
    parquet = hf_hub_download("wikiann", "en/test-00000-of-00001.parquet", repo_type="dataset")
    ds = load_dataset("parquet", data_files={"test": parquet}, split="test")
    return _write_jsonl(target, _bio_records_from_hf(ds, "ner_tags"))


def download_fewnerd_ner(path: str | Path, force: bool = False) -> int:
    """Download the FewNERD coarse-type NER gold (``bench_ner_fewnerd``).

    FewNERD is gated on the HuggingFace Hub: accept its license at
    ``https://huggingface.co/datasets/fewnerd`` and set ``HF_TOKEN``. Until
    then this raises ``RuntimeError``, like the other license-gated golds.
    """
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0
    from datasets import load_dataset

    try:
        ds = load_dataset("fewnerd", split="test")
    except Exception as exc:
        raise RuntimeError(
            "FewNERD is license-gated on the HuggingFace Hub: accept the license "
            "at https://huggingface.co/datasets/fewnerd, set HF_TOKEN, and retry. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return _write_jsonl(target, _bio_records_from_hf(ds, "coarse_ner_tags"))


def _bio_to_entities(text: str, tokens: list[str], labels: list[str]) -> dict[str, Any]:
    """Convert IOB2-tagged tokens to entity span dicts."""
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    char_pos = 0
    for token, label in zip(tokens, labels, strict=False):
        token_start = text.index(token, char_pos) if token in text[char_pos:] else char_pos
        char_pos = token_start + len(token)
        tag = label[2:] if len(label) > 2 and label[1] == "-" else ""
        is_begin = label.startswith("B-") or (
            label.startswith("I-") and (current is None or current["type"] != tag)
        )
        is_inside = label.startswith("I-") and current is not None and current["type"] == tag

        if is_begin:
            if current:
                entities.append(current)
            current = {"name": token, "type": tag, "start": token_start, "end": char_pos}
        elif is_inside:
            current["name"] += " " + token
            current["end"] = char_pos
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return {"text": text, "entities": entities}


# ═══════════════════════════════════════════════════════════════
# Chunking — derived from CoNLL-2003
# ═══════════════════════════════════════════════════════════════


def download_conll_chunking(path: str | Path, force: bool = False) -> int:
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        urlretrieve(CONLL_TRAIN_URL, tmp.name)

    records: list[dict[str, Any]] = []
    tokens: list[str] = []
    labels: list[str] = []

    with open(tmp.name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    records.append(_bio_to_chunks(tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split()
            if len(parts) >= 4:
                tokens.append(parts[0])
                labels.append(parts[3])
        if tokens:
            records.append(_bio_to_chunks(tokens, labels))

    os.unlink(tmp.name)
    return _write_jsonl(target, records)


def _bio_to_chunks(tokens: list[str], labels: list[str]) -> dict[str, Any]:
    """Extract gold entity mentions from IOB2 tags."""
    entities: list[dict[str, str]] = []
    current_name: list[str] = []
    current_type: str = ""
    for token, label in zip(tokens, labels, strict=False):
        tag = label[2:] if len(label) > 2 and label[1] == "-" else ""
        is_begin = label.startswith("B-") or (
            label.startswith("I-") and (not current_name or current_type != tag)
        )
        is_inside = label.startswith("I-") and current_name and current_type == tag

        if is_begin:
            if current_name:
                entities.append({"name": " ".join(current_name), "type": current_type})
            current_name = [token]
            current_type = tag
        elif is_inside:
            current_name.append(token)
        else:
            if current_name:
                entities.append({"name": " ".join(current_name), "type": current_type})
                current_name, current_type = [], ""
    if current_name:
        entities.append({"name": " ".join(current_name), "type": current_type})
    return {"text": " ".join(tokens), "chunks": entities}


# ═══════════════════════════════════════════════════════════════
# Dedup — DBLP-ACM
# ═══════════════════════════════════════════════════════════════


def download_dblp_dedup(path: str | Path, force: bool = False) -> int:
    """Download the DBLP-ACM dedup benchmark as ``{text_a, text_b, label}`` records.

    The two mirrors ship different layouts, both of which are handled:

    * **Deepmatcher layout** — a CSV with ``label``/``title_left``/``title_right``
      columns describing duplicate pairs directly.
    * **Uni-Leipzig layout** — separate ``DBLP2.csv`` and ``ACM.csv`` tables plus a
      ``DBLP-ACM_perfectMapping.csv`` of gold duplicate id pairs.  Positive pairs
      come from the mapping; a balanced, deterministic set of negative pairs is
      sampled from non-matching records.

    Args:
        path: Where to write the gold JSONL.
        force: Re-download even if *path* already has data.

    Returns:
        Number of gold records written.
    """
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0

    import tempfile
    from urllib.error import HTTPError, URLError

    zip_path: str | None = None
    for mirror in DBLP_ACM_MIRRORS:
        try:
            # Use a context-managed temp file; delete=False keeps the path
            # alive for reading after the handle closes.
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                urlretrieve(mirror, tmp.name)
            zip_path = tmp.name
            logger.info("Downloaded DBLP-ACM from %s", mirror)
            break
        except (HTTPError, URLError, OSError) as exc:
            logger.warning("Mirror failed (%s): %s", mirror, exc)
            continue

    if zip_path is None:
        raise RuntimeError(
            "Could not download DBLP-ACM from any mirror. Place the dataset "
            "manually at: " + str(target)
        )

    try:
        with zipfile.ZipFile(zip_path) as zf:
            records = _dblp_acm_records(zf)
    finally:
        os.unlink(zip_path)
    return _write_jsonl(target, records)


def _dblp_acm_records(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    """Convert a downloaded DBLP-ACM zip into ``{text_a, text_b, label}`` records."""
    import csv
    import io

    csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
    if not csv_names:
        raise ValueError("DBLP-ACM zip contains no CSV files")

    # Detect layout from the first CSV's header.
    with zf.open(csv_names[0]) as f:
        fields = (
            csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace")).fieldnames or []
        )
    if "title_left" in fields:
        return _dblp_acm_deepmatcher_pairs(zf)
    if any("perfectMapping" in name for name in csv_names):
        return _dblp_acm_unileipzig_pairs(zf)
    raise ValueError(
        "Unrecognized DBLP-ACM format. Expected deepmatcher pairs "
        "(title_left/title_right) or uni-leipzig tables + perfectMapping."
    )


def _dblp_acm_deepmatcher_pairs(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    """Read ``{text_a, text_b, label}`` pairs from the deepmatcher CSV layout."""
    import csv
    import io

    records: list[dict[str, Any]] = []
    for name in zf.namelist():
        if not name.endswith(".csv"):
            continue
        with zf.open(name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            if not reader.fieldnames or "title_left" not in reader.fieldnames:
                continue
            for row in reader:
                label = (
                    "duplicate"
                    if str(row.get("label", "")).lower() in ("1", "yes", "true")
                    else "not_duplicate"
                )
                records.append(
                    {
                        "text_a": f"{row.get('title_left', '')}. {row.get('authors_left', '')}.",
                        "text_b": f"{row.get('title_right', '')}. {row.get('authors_right', '')}.",
                        "label": label,
                    }
                )
    return records


def _dblp_acm_unileipzig_pairs(zf: zipfile.ZipFile) -> list[dict[str, Any]]:
    """Build ``{text_a, text_b, label}`` pairs from the uni-leipzig tables + mapping.

    The zip holds ``DBLP2.csv`` and ``ACM.csv`` (``id,title,authors,venue,year``)
    plus ``DBLP-ACM_perfectMapping.csv`` (``idDBLP,idACM``) listing gold duplicate
    pairs.  Every mapping pair becomes a positive example; a balanced set of
    deterministic negative pairs is sampled from non-matching records.
    """
    import csv
    import io
    import random

    tables: dict[str, dict[str, dict[str, str]]] = {}
    mapping: list[tuple[str, str]] = []
    for name in zf.namelist():
        if not name.endswith(".csv"):
            continue
        with zf.open(name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            if not reader.fieldnames:
                continue
            fields = set(reader.fieldnames)
            if "perfectMapping" in name:
                for row in reader:
                    mapping.append(
                        (
                            str(row.get("idDBLP", "")).strip(),
                            str(row.get("idACM", "")).strip(),
                        )
                    )
            elif {"id", "title", "authors"} <= fields:
                table: dict[str, dict[str, str]] = {}
                for row in reader:
                    table[str(row.get("id", "")).strip()] = {
                        "title": str(row.get("title", "")).strip(),
                        "authors": str(row.get("authors", "")).strip(),
                    }
                tables[name] = table

    dblp = next((t for n, t in tables.items() if "dblp" in n.lower()), None)
    acm = next((t for n, t in tables.items() if "acm" in n.lower()), None)
    if dblp is None or acm is None or not mapping:
        raise ValueError("Uni-Leipzig DBLP-ACM zip is missing tables or the perfect mapping.")

    def render(record: dict[str, str]) -> str:
        return f"{record['title']}. {record['authors']}."

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for dblp_id, acm_id in mapping:
        seen.add((dblp_id, acm_id))
        left = dblp.get(dblp_id)
        right = acm.get(acm_id)
        if left is None or right is None:
            continue
        records.append({"text_a": render(left), "text_b": render(right), "label": "duplicate"})

    # Deterministic negative pairs — never sample a gold duplicate pair.
    rng = random.Random(0)
    dblp_ids = list(dblp)
    acm_ids = list(acm)
    target = len(records)
    attempts = 0
    max_attempts = target * 20 + 100
    while len(records) < target * 2 and attempts < max_attempts:
        attempts += 1
        pair = (rng.choice(dblp_ids), rng.choice(acm_ids))
        if pair in seen:
            continue
        seen.add(pair)
        records.append(
            {
                "text_a": render(dblp[pair[0]]),
                "text_b": render(acm[pair[1]]),
                "label": "not_duplicate",
            }
        )
    return records


# ═══════════════════════════════════════════════════════════════
# Resolution — T2D
# ═══════════════════════════════════════════════════════════════


def download_t2d_resolution(path: str | Path, force: bool = False) -> int:
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0

    records = [
        {"names": ["United States", "USA", "U.S."], "cluster_id": "Q30"},
        {"names": ["China", "People's Republic of China"], "cluster_id": "Q148"},
    ]
    logger.warning("T2D requires manual download from webdatacommons.org. Using placeholders.")
    return _write_jsonl(target, records)


# ═══════════════════════════════════════════════════════════════
# Quality — TACRED
# ═══════════════════════════════════════════════════════════════


def download_tacred_quality(path: str | Path, force: bool = False) -> int:
    """Write a small placeholder quality-filter gold dataset.

    The real TACRED benchmark (https://catalog.ldc.upenn.edu/LDC2018T24) is
    LDC license-gated and cannot be auto-downloaded.  To exercise the
    benchmark end-to-end (the same pattern as the T2D resolution
    placeholders), this writes a tiny, clearly synthetic gold set of
    keep/reject texts.  Replace it with real TACRED records for meaningful
    numbers — each line is ``{"text": "...", "label": "keep"|"reject"}``.

    Args:
        path: Where to write the gold JSONL.
        force: Re-download even if *path* already has data.

    Returns:
        Number of gold records written.
    """
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0

    keep = [
        (
            "KGLab is an open-source library for building knowledge graphs "
            "from unstructured text. It chains configurable stages: loading raw "
            "documents, cleaning and normalizing the text, filtering low-quality "
            "content, deduplicating near-identical pages, and splitting the "
            "remainder into semantically coherent chunks. Each stage exposes a "
            "small registry of methods so researchers can swap implementations "
            "and compare results across pipelines."
        ),
        (
            "Entity resolution is the task of deciding whether two mentions refer "
            "to the same real-world entity. String methods compare surface forms "
            "directly, while embedding methods compare dense vector representations "
            "that capture semantic similarity. A good resolver keeps a knowledge "
            "graph compact by merging duplicates without collapsing genuinely "
            "distinct concepts into a single node."
        ),
        (
            "Retrieval-augmented generation grounds an LLM's answer in external "
            "knowledge. A knowledge graph can serve as that source: entities are "
            "linked to source chunks, so a query that matches an entity retrieves "
            "the evidence needed to answer accurately. This reduces hallucination "
            "and makes the model's reasoning traceable to concrete documents."
        ),
    ]
    reject = [
        "click here for more",
        "Sign in | Register | Help | Cart",
        "Privacy Policy Terms of Use About",
        "Top stories Opinion Sports Business",
        "Loading... please wait",
        "© 2026 Example Corp. All rights reserved.",
    ]

    records = [{"text": text, "label": "keep"} for text in keep]
    records.extend({"text": text, "label": "reject"} for text in reject)
    logger.warning(
        "TACRED is license-gated — writing %d synthetic placeholder records to %s. "
        "Replace with real data for meaningful quality scores.",
        len(records),
        target,
    )
    return _write_jsonl(target, records)


# ═══════════════════════════════════════════════════════════════
# RAG — HotpotQA
# ═══════════════════════════════════════════════════════════════


def download_hotpotqa_rag(path: str | Path, force: bool = False) -> int:
    """Download HotpotQA train as ``{query, answer_entity, supporting_chunks}`` gold.

    HotpotQA's canonical host (curtis.ml.cmu.edu) is frequently offline or
    extremely slow.  A socket timeout keeps this from hanging; if the
    download fails, place a gold JSONL manually at *path* with records::

        {"query": "...", "answer_entity": "...", "supporting_chunks": ["..."]}

    Args:
        path: Where to write the gold JSONL.
        force: Re-download even if *path* already has data.

    Returns:
        Number of gold records written.

    Raises:
        RuntimeError: If the download fails — the message includes the path
            and schema needed for manual placement.
    """
    target = Path(path)
    if _skip_if_cached(target, force):
        return 0

    import socket
    import tempfile

    socket.setdefaulttimeout(30)
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            urlretrieve(HOTPOTQA_URL, tmp.name)
    except OSError as exc:
        raise RuntimeError(
            "Could not download HotpotQA from "
            + HOTPOTQA_URL
            + f" ({exc}). The canonical host is often offline. Place a gold "
            + "JSONL at: "
            + str(target)
            + " with one record per line: {"
            + '"query": ..., "answer_entity": ..., "supporting_chunks": [...]}'
        ) from exc

    with open(tmp.name, encoding="utf-8") as f:
        data = json.load(f)
    os.unlink(tmp.name)

    records: list[dict[str, Any]] = []
    for item in data:
        facts = item.get("supporting_facts", [])
        ctx_titles = {t: i for i, (t, _) in enumerate(item.get("context", []))}
        chunks: list[str] = []
        for title, sent_idx in facts:
            if title in ctx_titles:
                ctx = item["context"][ctx_titles[title]]
                if sent_idx < len(ctx[1]):
                    chunks.append(ctx[1][sent_idx])
        records.append(
            {
                "query": item.get("question", ""),
                "answer_entity": item.get("answer", ""),
                "supporting_chunks": chunks,
                "type": item.get("type", ""),
                "level": item.get("level", ""),
            }
        )
    return _write_jsonl(target, records)
