"""Entity linking contracts — match extracted entities to external knowledge bases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EntityLinker(ABC):
    """Match extracted entities to entries in an external knowledge base.

    Subclass and implement ``link()`` to connect to Wikidata, DBpedia,
    a custom taxonomy, or any lookup table.  Return an enriched entity
    dict (same keys plus ``kb_id``, ``kb_source``, etc.) or ``None`` if
    no match was found.

    Usage::

        class MyLinker(EntityLinker):
            def link(self, entity):
                match = my_lookup(entity["name"])
                if match:
                    return {**entity, "kb_id": match["id"], "kb_source": "my_db"}
                return None
    """

    @abstractmethod
    def link(self, entity: dict[str, Any]) -> dict[str, Any] | None:
        """Attempt to link a single entity to an external KB.

        Args:
            entity: Dict with at least ``"name"``, ``"type"``, and ``"id"`` keys.

        Returns:
            The entity dict enriched with linking metadata (``kb_id``,
            ``kb_source``), or ``None`` if no match was found.
        """
        ...
