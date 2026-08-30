"""Series-based document-to-document relation extraction."""

import logging
from typing import Any

from kglab._shared import Document
from kglab.kg_build.extract.document_relation._base import (
    DocTriple,
    DocumentRelationExtractor,
)
from kglab.kg_build.extract.document_relation._helpers import (
    doc_entity_id,
    make_document_entities,
)

logger = logging.getLogger(__name__)


class SeriesExtractor(DocumentRelationExtractor):
    """Extract ``is_part_of_series`` relations by matching series metadata
    across documents.
    """

    def extract(self, documents: list[Document]) -> tuple[list[dict[str, Any]], list[DocTriple]]:
        doc_entities = make_document_entities(documents)
        series_groups: dict[str, list[tuple[Document, str]]] = {}
        for doc in documents:
            series = doc.metadata.get("series")
            if series:
                key = str(series).strip().casefold()
                series_groups.setdefault(key, []).append((doc, doc_entity_id(doc)))

        triples: list[DocTriple] = []
        for series_id, group in series_groups.items():
            if len(group) < 2:
                continue
            ordered = self._try_order(group)
            if ordered:
                for k in range(len(ordered) - 1):
                    triples.append(
                        (
                            ordered[k][1],
                            "is_part_of_series",
                            ordered[k + 1][1],
                            series_id,
                            "",
                        )
                    )
            else:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        triples.append(
                            (
                                group[i][1],
                                "is_part_of_series",
                                group[j][1],
                                series_id,
                                "",
                            )
                        )

        logger.info(
            "[SeriesExtractor] %d series found, %d edges",
            len(series_groups),
            len(triples),
        )
        return doc_entities, triples

    @staticmethod
    def _try_order(
        group: list[tuple[Document, str]],
    ) -> list[tuple[Document, str]] | None:
        has_index = any("series_index" in doc.metadata for doc, _ in group)
        if has_index:
            return sorted(
                group,
                key=lambda item: int(item[0].metadata.get("series_index", 0)),
            )
        has_date = any(
            "upload_date" in doc.metadata or "publish_date" in doc.metadata for doc, _ in group
        )
        if has_date:

            def _date_key(item: tuple[Document, str]) -> str:
                return str(
                    item[0].metadata.get("upload_date")
                    or item[0].metadata.get("publish_date")
                    or ""
                )

            return sorted(group, key=_date_key)
        return None
