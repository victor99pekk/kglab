"""Entity resolution methods.

Available methods:
    string     — Name-based token overlap resolution (fast, no deps)
    embedding  — Semantic similarity clustering (sentence-transformers)
"""

from .embedding import resolve_embedding
from .string import resolve_string


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
        if method not in {"string", "embedding"}:
            raise ValueError("Entity resolution method must be one of: string, embedding")
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


__all__ = ["EntityResolver", "resolve_string", "resolve_embedding"]
