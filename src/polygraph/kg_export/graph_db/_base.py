"""Abstract base for graph database uploaders.

To add a new backend, subclass ``GraphDBUploader`` and implement ``upload()``,
then register it in ``graph_db.BACKENDS``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class GraphDBUploader(ABC):
    """Upload a knowledge graph JSON file to a graph database.

    Subclass and implement ``upload()``. The constructor takes connection
    parameters (host, credentials) — each backend defines its own signature.

    Usage::

        class MyBackend(GraphDBUploader):
            def __init__(self, host: str, **kwargs) -> None:
                self.host = host

            def upload(self, json_path: str | Path, clear: bool = False) -> None:
                ...  # connect to MyBackend, load JSON, write graph

        from polygraph.kg_export.graph_db import BACKENDS
        BACKENDS["my_backend"] = MyBackend
    """

    @abstractmethod
    def upload(self, json_path: str | Path, clear: bool = False) -> None:
        """Load a KG JSON file into the graph database.

        Args:
            json_path: Path to a ``knowledge_graph.json`` file.
            clear: If True, wipe the database before uploading.
        """
        ...
