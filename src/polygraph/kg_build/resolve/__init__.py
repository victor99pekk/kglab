"""Entity resolution methods.

Available methods:
    string     — Name-based token overlap resolution (fast, no deps)
    embedding  — Semantic similarity clustering (sentence-transformers)
"""

from .embedding import resolve_embedding
from .registry import RESOLUTION_METHODS, get_resolution_method
from .string import resolve_string


def with_method(entities, *, method: str = "string", **options):
    """Resolve entities with a registered method."""
    return get_resolution_method(method)(entities, **options)


def by_string(entities, threshold: float = 0.85):
    return with_method(entities, method="string", threshold=threshold)


def by_embedding(
    entities,
    threshold: float = 0.85,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    encoder=None,
):
    return with_method(
        entities,
        method="embedding",
        threshold=threshold,
        model_name=model_name,
        encoder=encoder,
    )


# Backward compatibility: EntityResolver class delegates to the new functions
class EntityResolver:
    """Resolves duplicate entities — delegates to string or embedding methods."""

    def __init__(
        self,
        threshold: float = 0.80,
        method: str = "embedding",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        encoder=None,
    ) -> None:
        if method == "string_similarity":
            method = "string"
        if method not in RESOLUTION_METHODS:
            choices = ", ".join(sorted(RESOLUTION_METHODS))
            raise ValueError(f"Entity resolution method must be one of: {choices}")
        self.threshold = threshold
        self.method = method
        self.model_name = model_name
        self.encoder = encoder

    def resolve(self, entities):
        if not entities:
            return []
        if self.method == "embedding":
            return resolve_embedding(entities, self.threshold, self.model_name, self.encoder)
        return resolve_string(entities, self.threshold)

    def resolve_with_mapping(self, entities):
        resolved = self.resolve(entities)
        alias_index = {}
        for canonical in resolved:
            canonical_id = canonical.get("id", "")
            entity_type = canonical.get("type", canonical.get("label", "ENTITY"))
            names = [canonical.get("name", ""), *canonical.get("aliases", [])]
            for name in names:
                if name:
                    alias_index[(entity_type, str(name).casefold().strip())] = canonical_id
        id_map = {}
        for entity in entities:
            original_id = entity.get("id", "")
            entity_type = entity.get("type", entity.get("label", "ENTITY"))
            canonical_id = alias_index.get(
                (entity_type, str(entity.get("name", "")).casefold().strip())
            )
            if original_id and canonical_id:
                id_map[original_id] = canonical_id
        return resolved, id_map


__all__ = [
    "EntityResolver",
    "RESOLUTION_METHODS",
    "by_embedding",
    "by_string",
    "get_resolution_method",
    "resolve_embedding",
    "resolve_string",
    "with_method",
]
