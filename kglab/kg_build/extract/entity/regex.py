"""Dependency-free regular-expression entity extraction method."""

import logging
import re

from kglab.kg_build.extract._base import Entity, EntityExtractor

logger = logging.getLogger(__name__)


class SimpleExtractor(EntityExtractor):
    """Extract emails, URLs, and capitalized phrases with regular expressions."""

    CAPITALIZED_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    EMAIL = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
    URL = re.compile(r"https?://[^\s]+")

    def extract(self, text: str) -> list[Entity]:
        entities: list[Entity] = []
        seen: set[str] = set()

        for pattern, label, confidence in (
            (self.EMAIL, "CONTACT", 1.0),
            (self.URL, "URL", 1.0),
            (self.CAPITALIZED_PHRASE, "NAMED_ENTITY", 0.50),
        ):
            for match in pattern.finditer(text):
                name = match.group()
                if name.casefold() in seen:
                    continue
                seen.add(name.casefold())
                entities.append(
                    Entity(
                        name=name,
                        label=label,
                        mentions=[name],
                        confidence=confidence,
                    )
                )

        logger.debug("SimpleExtractor: found %d entities", len(entities))
        return entities
