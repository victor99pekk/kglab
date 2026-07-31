"""spaCy English entity extraction method."""

import logging

from polygraph.kg_build.extract._base import Entity, EntityExtractor

logger = logging.getLogger(__name__)


class SpacyExtractor(EntityExtractor):
    """Extract named entities and noun-phrase concepts with spaCy (English)."""

    def __init__(self, model_name: str = "en_core_web_lg") -> None:
        self.model_name = model_name
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load(self.model_name)
            except OSError:
                logger.warning(
                    "spaCy model '%s' not found. Install with: python -m spacy download %s",
                    self.model_name,
                    self.model_name,
                )
                import spacy

                self._nlp = spacy.blank("en")
        return self._nlp

    def extract(self, text: str) -> list[Entity]:
        doc = self.nlp(text)
        entities: list[Entity] = []
        seen: set[str] = set()

        for ent in doc.ents:
            name = ent.text.strip()
            if name.casefold() not in seen and len(name) > 1:
                seen.add(name.casefold())
                entities.append(
                    Entity(
                        name=name,
                        label=ent.label_,
                        mentions=[name],
                        confidence=0.90,
                    )
                )

        try:
            for chunk in doc.noun_chunks:
                name = chunk.text.strip()
                if name.casefold() not in seen and len(name) > 2:
                    seen.add(name.casefold())
                    entities.append(
                        Entity(
                            name=name,
                            label="CONCEPT",
                            mentions=[name],
                            confidence=0.80,
                        )
                    )
        except Exception:
            logger.debug("Noun chunk extraction skipped (parser may not be available)")

        return entities
