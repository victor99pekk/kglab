"""Baseline data loading from files and directories.

Exports: DataLoader
"""

import csv
import json
import logging
from pathlib import Path

from polygraph._shared import Document

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads text data from files and directories."""

    def __init__(self, supported_formats: list[str] | None = None) -> None:
        self.supported = supported_formats or ["txt", "json", "csv", "jsonl"]

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
            return [
                Document(
                    content=item.get("text", item.get("content", "")),
                    source=str(path),
                    doc_id=item.get("url") or item.get("id") or f"{path}#{i}",
                    metadata={k: v for k, v in item.items() if k not in ("text", "content", "id")},
                )
                for i, item in enumerate(data)
            ]
        return [
            Document(
                content=data.get("text", data.get("content", "")),
                source=str(path),
                doc_id=data.get("url") or data.get("id") or str(path),
            )
        ]

    def _load_jsonl(self, path: Path) -> list[Document]:
        docs = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    doc_id = item.get("url") or item.get("id") or f"{path}#{i}"
                    metadata = {k: v for k, v in item.items() if k not in ("text", "content", "id")}
                    metadata.setdefault("upload_date", "")
                    docs.append(
                        Document(
                            content=item.get("text", item.get("content", "")),
                            source=str(path),
                            doc_id=doc_id,
                            metadata=metadata,
                        )
                    )
        return docs

    def _load_csv(self, path: Path) -> list[Document]:
        docs = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text_col = next(
                    (k for k in row if k.lower() in ("text", "content", "body")),
                    list(row.keys())[0],
                )
                docs.append(
                    Document(
                        content=row[text_col],
                        source=str(path),
                        doc_id=row.get("url") or row.get("id") or f"{path}#{i}",
                        metadata={k: v for k, v in row.items() if k != text_col and k != "id"},
                    )
                )
        return docs
