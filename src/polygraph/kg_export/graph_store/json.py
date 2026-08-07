"""File-backed graph stores for exported JSON / GraphML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polygraph.kg_export.graph_store.networkx import NetworkXGraphStore


class JSONGraphStore(NetworkXGraphStore):
    """``GraphStore`` reading a ``knowledge_graph.json`` export.

    The file is parsed lazily on first access into an in-memory graph.  The
    ``metadata``, ``entities``, and ``triples`` payloads are also exposed.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._raw: dict[str, Any] | None = None
        super().__init__(loader=self._load)

    def _load(self) -> Any:
        import networkx as nx

        with open(self._path, encoding="utf-8") as fh:
            self._raw = json.load(fh)
        return nx.node_link_graph(self._raw.get("graph", {}))

    # ── Extra accessors for the JSON payload ────────────────────

    def _ensure_loaded(self) -> None:
        """Force the lazy file load so ``_raw`` is populated."""
        _ = self.graph

    @property
    def metadata(self) -> dict[str, Any]:
        """The ``metadata`` block from the JSON export."""
        self._ensure_loaded()
        return (self._raw or {}).get("metadata", {})

    @property
    def entities(self) -> list[dict[str, Any]]:
        """The ``entities`` list from the JSON export."""
        self._ensure_loaded()
        return (self._raw or {}).get("entities", [])

    @property
    def triples(self) -> list[dict[str, Any]]:
        """The ``triples`` list from the JSON export."""
        self._ensure_loaded()
        return (self._raw or {}).get("triples", [])


class GraphMLGraphStore(NetworkXGraphStore):
    """``GraphStore`` reading a ``.graphml`` export (lazily)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        super().__init__(loader=self._load)

    def _load(self) -> Any:
        import networkx as nx

        return nx.read_graphml(self._path)


__all__ = ["GraphMLGraphStore", "JSONGraphStore"]
