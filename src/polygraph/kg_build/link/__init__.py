"""Entity linking — match extracted entities to external knowledge bases.

Convenience API::

    from polygraph.kg_build import link

    linked = link.entities(resolved_entities, method="wikidata")

Class API::

    from polygraph.kg_build.link import WikidataLinker
    linker = WikidataLinker()
    result = linker.link(entity)
"""

from __future__ import annotations

from typing import Any

from polygraph.kg_build.link._base import EntityLinker
from polygraph.kg_build.link.registry import LINKER_REGISTRY, create_linker


def entities(
    entity_list: list[dict[str, Any]],
    method: str,
    **options: Any,
) -> list[dict[str, Any]]:
    """Link a batch of entities to an external KB.

    Args:
        entity_list: Resolved entities (each with ``name``, ``type``, ``id``).
        method: Registered linker name (required — add entries to
            ``LINKER_REGISTRY`` for your implementation).
        **options: Forwarded to the linker constructor.

    Returns:
        The entity list with linked entities enriched (``kb_id``,
        ``kb_source``).  Unmatched entities are passed through unchanged.
    """
    linker = create_linker(method, **options)
    linked_count = 0
    result: list[dict[str, Any]] = []
    for entity in entity_list:
        enriched = linker.link(entity)
        if enriched and "kb_id" in enriched:
            linked_count += 1
            result.append(enriched)
        else:
            result.append(entity)

    print(f"[link] {linked_count}/{len(entity_list)} entities linked to {method}")
    return result


__all__ = [
    "EntityLinker",
    "LINKER_REGISTRY",
    "create_linker",
    "entities",
]
