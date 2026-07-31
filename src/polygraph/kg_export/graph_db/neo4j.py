"""Neo4j graph database uploader."""

from __future__ import annotations

import logging
from pathlib import Path

from polygraph.kg_export.graph_db._base import GraphDBUploader

logger = logging.getLogger(__name__)


class Neo4jUploader(GraphDBUploader):
    """Upload a knowledge graph to Neo4j.

    Connection is configured via constructor arguments or environment variables.

    Args:
        uri: Neo4j bolt URI (default: ``NEO4J_URI`` env var or ``bolt://localhost:7687``).
        user: Neo4j username (default: ``NEO4J_USER`` env var or ``neo4j``).
        password: Neo4j password (default: ``NEO4J_PASSWORD`` env var).

    Usage::

        uploader = Neo4jUploader(uri="bolt://localhost:7687", user="neo4j", password="secret")
        uploader.upload("output/knowledge_graph.json", clear=True)
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        import os

        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")

    def upload(self, json_path: str | Path, clear: bool = False) -> None:
        """Load a KG JSON file into Neo4j.

        Args:
            json_path: Path to ``knowledge_graph.json``.
            clear: If True, wipe the database before uploading.
        """
        from polygraph.kg_export.neo4j.upload import upload_graph

        upload_graph(
            str(json_path), clear=clear, uri=self.uri, user=self.user, password=self.password
        )
