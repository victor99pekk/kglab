"""String-based entity resolution — token overlap without embeddings.

Exports: resolve_string, EntityResolver (for class API)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _string_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity between two strings."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def resolve_string(entities: list[dict[str, Any]], threshold: float = 0.85) -> list[dict[str, Any]]:
    """Resolve duplicate entities by normalized string similarity."""
    if not entities:
        return []

    resolved: list[dict[str, Any]] = []
    seen: list[str] = []

    for entity in entities:
        name = entity["name"].lower().strip()
        matched = False

        for i, existing in enumerate(seen):
            existing_type = resolved[i].get("type", resolved[i].get("label", "ENTITY"))
            entity_type = entity.get("type", entity.get("label", "ENTITY"))
            if existing_type != entity_type:
                continue
            if _string_similarity(name, existing) >= threshold:
                canonical = resolved[i]
                merged_aliases = set(canonical.get("aliases", []))
                merged_aliases.update(entity.get("aliases", []))
                merged_aliases.add(entity.get("name", ""))
                canonical["aliases"] = sorted(a for a in merged_aliases if a)
                canonical["confidenceScore"] = max(
                    canonical.get("confidenceScore", 0), entity.get("confidenceScore", 0)
                )
                if len(entity.get("description", "")) > len(canonical.get("description", "")):
                    canonical["description"] = entity["description"]
                existing_src = canonical.get("source", [])
                if not isinstance(existing_src, list):
                    existing_src = [existing_src] if existing_src else []
                new_src = entity.get("source", [])
                if isinstance(new_src, str):
                    new_src = [new_src] if new_src else []
                canonical["source"] = list(dict.fromkeys(existing_src + new_src))
                existing_chunks = canonical.get("source_chunk_ids", [])
                if not isinstance(existing_chunks, list):
                    existing_chunks = [existing_chunks] if existing_chunks else []
                new_chunks = entity.get("source_chunk_ids", [])
                if not isinstance(new_chunks, list):
                    new_chunks = [new_chunks] if new_chunks else []
                canonical["source_chunk_ids"] = list(dict.fromkeys(existing_chunks + new_chunks))
                matched = True
                break

        if not matched:
            seen.append(name)
            resolved.append(dict(entity))

    logger.info(f"String resolution: {len(entities)} -> {len(resolved)} unique entities")
    return resolved
