"""Baseline data loading from files and directories.

Exports: DataLoader
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from polygraph._shared import Document

logger = logging.getLogger(__name__)

_DEFAULT_FIELD_MAPPING: dict[str, str] = {
    "text": "text",
    "id": "id",
    "url": "url",
    "title": "title",
}


class DataLoader:
    """Loads text data from files and directories.

    Args:
        supported_formats: File extensions to scan for in directories.
        field_mapping: Map canonical field names to user-defined field names
            in the source data. Example: ``{"text": "body", "id": "doc_id"}``
            means source records use ``body`` and ``doc_id`` instead of
            ``text`` and ``id``. Unknown fields are passed through to
            ``Document.metadata`` unchanged.
    """

    def __init__(
        self,
        supported_formats: list[str] | None = None,
        field_mapping: dict[str, str] | None = None,
    ) -> None:
        self.supported = supported_formats or ["txt", "json", "csv", "jsonl"]
        self._mapping: dict[str, str] = {**_DEFAULT_FIELD_MAPPING, **(field_mapping or {})}

    def _field(self, name: str) -> str:
        """Return the user-facing field name for a canonical field."""
        return self._mapping.get(name, name)

    def load(self, paths: list[Path]) -> list[Document]:
        """Load documents from a list of file/directory paths."""
        documents: list[Document] = []

        for path in paths:
            if path.is_dir():
                for ext in self.supported:
                    for file_path in path.rglob(f"*.{ext}"):
                        documents.extend(self._load_file(file_path))
            else:
                documents.extend(self._load_file(path))

        logger.info(f"Loaded {len(documents)} documents from {len(paths)} path(s)")
        return documents

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return self._load_txt(path)
        elif suffix == ".json":
            return self._load_json(path)
        elif suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix == ".csv":
            return self._load_csv(path)
        else:
            logger.warning(f"Unsupported format: {suffix} — skipping {path}")
            return []

    def _load_txt(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
        return [Document(content=text, source=str(path), doc_id=str(path))]

    def _load_json(self, path: Path) -> list[Document]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [self._make_doc(item, path, i) for i, item in enumerate(data)]
        return [self._make_doc(data, path, 0)]

    def _load_jsonl(self, path: Path) -> list[Document]:
        docs = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    docs.append(self._make_doc(json.loads(line), path, i))
        return docs

    def _load_csv(self, path: Path) -> list[Document]:
        docs = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                docs.append(self._make_doc(row, path, i))
        return docs

    def _make_doc(self, item: dict, path: Path, index: int) -> Document:
        """Build a Document from a parsed record using the field mapping."""
        text_key = self._field("text")
        id_key = self._field("id")
        url_key = self._field("url")

        content = str(item.get(text_key, item.get("content", "")))
        doc_id = str(item.get(url_key) or item.get(id_key) or f"{path}#{index}")

        # All fields except content source fields go into metadata
        skip_keys = {text_key, id_key, "content"}
        metadata: dict[str, object] = {}
        for k, v in item.items():
            if k not in skip_keys:
                metadata[k] = v
        metadata.setdefault("upload_date", "")

        # Ensure canonical keys exist in metadata for downstream consumers.
        # title and url are optional — auto-generate fallbacks when missing.
        title_key = self._field("title")
        metadata["title"] = item.get(title_key) or doc_id
        metadata["url"] = item.get(url_key) or f"polygraph://{doc_id}"

        return Document(
            content=content,
            source=str(path),
            doc_id=doc_id,
            language=item.get("language", item.get("lang", "en")),
            metadata=metadata,
        )
