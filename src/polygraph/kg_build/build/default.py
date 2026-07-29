"""Default NetworkX graph-construction function."""

from typing import Any

import networkx as nx

from polygraph.kg_build.build import GraphBuilder


def build_graph(
    entities: list[dict[str, Any]],
    triples: list[tuple[str, ...]],
) -> nx.DiGraph:
    """Build the default directed graph representation."""
    return GraphBuilder().build(entities, triples)


__all__ = ["build_graph"]
