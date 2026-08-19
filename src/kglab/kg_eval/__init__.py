"""KG Evaluation: measure quality, structure, coverage.

Function API:
    from kglab.kg_eval import metrics, structural, evaluate_kg

    report = metrics.evaluate(graph, entities, triples)
    audit = structural.run(graph, entities, triples, ontology_path="configs/default_ontology.yaml")
    report = evaluate_kg(kg, output_dir="output/")  # one-shot quality + structural

Class API:
    from kglab.kg_eval import BaseEvaluator, QualityEvaluator, StructuralAuditor, AccuracyEvaluator
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kglab.kg_eval._base import BaseEvaluator
from kglab.kg_eval.metrics import AccuracyEvaluator, QualityEvaluator
from kglab.kg_eval.structural import StructuralAuditor
from kglab.kg_export.graph_store import Neo4jGraphStore

# ── Function API ────────────────────────────────────────────────

metrics = SimpleNamespace()
metrics.evaluate = lambda graph, entities, triples: (
    QualityEvaluator().evaluate(graph, entities, triples)
)
metrics.completeness = lambda entities: QualityEvaluator().completeness(entities)

structural = SimpleNamespace()
structural.run = lambda graph, entities, triples, ontology_path=None: (
    StructuralAuditor(ontology_path=ontology_path).evaluate(graph, entities, triples)
)


def evaluate_kg(
    kg: dict[str, Any] | None,
    output_dir: str | Path | None = None,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Evaluate a built KG — the standalone replacement for ``Pipeline.evaluate``.

    Runs quality metrics plus a structural audit, and (when ``llm_client``
    is provided) LLM accuracy evaluation.  With ``output_dir``, writes
    ``metrics.json``.

    Pipelines with no in-memory graph (e.g. Neo4j-backed streaming
    pipelines) have nothing to score here; if they already wrote
    ``metrics.json`` themselves, it is left untouched.

    Args:
        kg: Dict with ``graph``, ``entities``, ``triples`` keys (as returned
            by ``Pipeline.execute`` / ``build_kg``).
        output_dir: If given, write ``metrics.json`` here.
        llm_client: Optional ``(prompt: str) -> str`` callable for accuracy eval.

    Returns:
        The evaluation report dict.
    """
    graph = (kg or {}).get("graph")
    entities = (kg or {}).get("entities", [])
    triples = (kg or {}).get("triples", [])

    # Fall back to materialising a GraphStore when the KG dict has no
    # nx.DiGraph.  Neo4j stores are NOT materialised (that would defeat
    # streaming) — they fall through to the stats-only branch below.
    if graph is None:
        store = (kg or {}).get("graph_store")
        if store is not None and not isinstance(store, Neo4jGraphStore):
            graph = store.to_networkx()

    if graph is not None:
        report: dict[str, Any] = {}
        report.update(metrics.evaluate(graph, entities, triples))
        report["structural_audit"] = structural.run(graph, entities, triples)
        report["num_entities"] = len(entities)
        if llm_client is not None:
            report["accuracy"] = AccuracyEvaluator(llm_client=llm_client).evaluate(
                graph, entities, triples
            )
    else:
        # No in-memory graph to score (e.g. streaming pipelines in Neo4j).
        stats = (kg or {}).get("neo4j_stats", {})
        report = {
            "num_nodes": stats.get("nodes_written", 0),
            "num_edges": stats.get("edges_written", 0),
            "num_entities": len(entities),
            "num_triples": len(triples),
        }

    if output_dir is not None:
        path = Path(output_dir) / "metrics.json"
        if graph is None and path.exists():
            # Respect a metrics.json the pipeline wrote itself.
            return report
        path.write_text(json.dumps(report, indent=2, default=str))
    return report


__all__ = [
    "AccuracyEvaluator",
    "BaseEvaluator",
    "QualityEvaluator",
    "StructuralAuditor",
    "evaluate_kg",
    "metrics",
    "structural",
]
