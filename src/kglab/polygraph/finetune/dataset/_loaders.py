"""Knowledge graph and raw document loaders for dataset generation.

Load previously exported KGs back into memory for fine-tuning ablation studies.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx

from .en._templates import _STRUCTURAL_PREDICATES


def load_kg(
    path: Path,
) -> tuple[
    nx.DiGraph,
    list[dict[str, Any]],
    list[tuple[str, str, str, str, str]],
]:
    """Load a knowledge graph from a JSON export file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    triples_raw = data.get("triples", [])
    graph_data = data.get("graph", {})
    graph = nx.node_link_graph(graph_data, edges="edges")

    # Normalize to (subj, pred, obj, source_text, source_chunk_id).
    triples: list[tuple[str, str, str, str, str]] = []
    for t in triples_raw:
        if isinstance(t, dict):
            subj = str(t.get("subject", ""))
            pred = str(t.get("predicate", ""))
            obj = str(t.get("object", ""))
            source_text = str(t.get("evidence_sentence") or t.get("description") or "")
            source_chunk_value = t.get("source_chunk_id") or t.get("source_chunk_ids") or ""
            if isinstance(source_chunk_value, list | tuple | set):
                source_chunk_value = next(iter(source_chunk_value), "")
            source_chunk_id = str(source_chunk_value)
            if subj and obj and pred.upper() not in _STRUCTURAL_PREDICATES:
                triples.append((subj, pred, obj, source_text, source_chunk_id))
        elif isinstance(t, list | tuple):
            if len(t) >= 4:
                if str(t[1]).upper() not in _STRUCTURAL_PREDICATES:
                    triples.append(
                        (
                            str(t[0]),
                            str(t[1]),
                            str(t[2]),
                            str(t[3]),
                            str(t[4]) if len(t) > 4 else "",
                        )
                    )
            elif len(t) == 3 and str(t[1]).upper() not in _STRUCTURAL_PREDICATES:
                triples.append((str(t[0]), str(t[1]), str(t[2]), "", ""))

    return graph, entities, triples


def load_raw_documents_from_kg(path: Path) -> list[dict[str, str]]:
    """Recover cleaned source chunks embedded in a KG export.

    Model C receives these texts only. It does not consume entities, triples,
    relations, or graph paths, while still seeing the same processed corpus as
    Model B.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    documents: list[dict[str, str]] = []
    for node in data.get("graph", {}).get("nodes", []):
        if node.get("type") != "Chunk":
            continue
        content = str(node.get("text", "")).strip()
        if not content:
            continue
        documents.append(
            {
                "content": content,
                "source": str(node.get("id", node.get("source", "unknown"))),
            }
        )
    return documents


def load_raw_documents(paths: list[Path]) -> list[dict[str, str]]:
    """Load raw text documents from files/directories.

    Supports .txt, .jsonl (one JSON object per line, reads "text" or "content" field),
    and .json (array of objects or single object with "text"/"content").
    """
    docs: list[dict[str, str]] = []
    for path in paths:
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix in (".txt", ".jsonl", ".json"):
                    docs.extend(_load_single_document_file(file_path))
        elif path.suffix in (".txt", ".jsonl", ".json"):
            docs.extend(_load_single_document_file(path))
    return docs


def _load_single_document_file(path: Path) -> list[dict[str, str]]:
    """Load documents from a single file, dispatching by extension."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return [{"content": path.read_text(encoding="utf-8"), "source": str(path)}]

    if suffix == ".jsonl":
        docs: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = obj.get("text", obj.get("content", ""))
                if content:
                    docs.append({"content": content, "source": str(path)})
        return docs

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [
                {"content": item.get("text", item.get("content", "")), "source": str(path)}
                for item in data
                if item.get("text") or item.get("content")
            ]
        content = data.get("text", data.get("content", ""))
        if content:
            return [{"content": content, "source": str(path)}]
        return []

    return []
