"""Embedding-based entity resolution — semantic similarity clustering.

Exports: resolve_embedding
"""

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _string_similarity(a: str, b: str) -> float:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _acronym(name: str) -> str:
    return "".join(w[0] for w in re.findall(r"\b\w", name))


def _compact(name: str) -> str:
    return re.sub(r"\W", "", name).casefold()


def _surface_compatible(first: str, second: str) -> bool:
    first_key = first.casefold().strip()
    second_key = second.casefold().strip()
    if first_key == second_key or _string_similarity(first_key, second_key) > 0:
        return True
    return _acronym(first_key) == _compact(second_key) or (
        _acronym(second_key) == _compact(first_key)
    )


def _merge_cluster(entities: list[dict[str, Any]], cluster: list[int]) -> dict[str, Any]:
    cluster_entities = [entities[i] for i in cluster]
    best = max(
        cluster_entities,
        key=lambda e: (e.get("confidenceScore", 0), len(e.get("name", ""))),
    )
    all_aliases: list[str] = []
    all_sources: list[str] = []
    best_description = ""
    for e in cluster_entities:
        all_aliases.extend(e.get("aliases", []))
        all_aliases.append(e.get("name", ""))
        if e.get("source"):
            src = e["source"]
            if isinstance(src, list):
                all_sources.extend(src)
            else:
                all_sources.append(src)
        if len(e.get("description", "")) > len(best_description):
            best_description = e["description"]
    return {
        "id": best.get("id", f"entity:{best['name'].lower().replace(' ', '_')}"),
        "name": best["name"],
        "type": best.get("type", "ENTITY"),
        "aliases": sorted({a for a in all_aliases if a}),
        "description": best_description,
        "confidenceScore": max(e.get("confidenceScore", 0) for e in cluster_entities),
        "importanceScore": max(e.get("importanceScore", 0) for e in cluster_entities),
        "source": list(dict.fromkeys(all_sources)),
        "embedding": best.get("embedding"),
        "updatedAt": max((e.get("updatedAt", "") for e in cluster_entities), default=""),
    }


def resolve_embedding(
    entities: list[dict[str, Any]],
    threshold: float = 0.85,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    encoder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve duplicate entities by embedding similarity clustering."""
    if not entities:
        return []

    names = [e["name"] for e in entities]
    if encoder is not None:
        embeddings = encoder(names)
    else:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers unavailable — falling back to string matching")
            from polygraph.kg_build.resolve.string import resolve_string

            return resolve_string(entities, threshold)
        model = SentenceTransformer(model_name)
        embeddings = model.encode(names, show_progress_bar=False)

    clusters: list[list[int]] = []
    assigned: set[int] = set()

    for i in range(len(names)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(names)):
            if j in assigned:
                continue
            first_type = entities[i].get("type", entities[i].get("label", "ENTITY"))
            second_type = entities[j].get("type", entities[j].get("label", "ENTITY"))
            if first_type != second_type:
                continue
            if not _surface_compatible(names[i], names[j]):
                continue
            sim = float(
                np.dot(embeddings[i], embeddings[j])
                / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            )
            if sim >= threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    resolved: list[dict[str, Any]] = []
    for cluster in clusters:
        resolved.append(_merge_cluster(entities, cluster))

    logger.info(f"Embedding resolution: {len(entities)} -> {len(resolved)} unique entities")
    return resolved
