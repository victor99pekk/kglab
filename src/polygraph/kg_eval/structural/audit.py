"""
Method 1, Step 1 — Intrinsic Structural Audit

Deep-dive graph health check focused on "graph rot" that would poison
SFT training data. Goes beyond basic metrics (completeness, consistency)
to catch issues specific to LLM training data quality.

Checks:
  - Orphan rate (nodes with zero connections)
  - Graph density (too sparse or too dense?)
  - Schema/ontology compliance (do edges follow defined rules?)
  - Entity duplication (unmerged near-duplicate nodes via embeddings)
  - Multi-hop connectivity (can the graph support chain reasoning?)
"""

import json
import logging
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from polygraph.kg_eval._base import BaseEvaluator

logger = logging.getLogger(__name__)


class StructuralAuditor(BaseEvaluator):
    """Audits a knowledge graph for structural issues that would degrade SFT quality."""

    def __init__(
        self,
        ontology_path: Path | None = None,
        entity_dedup_threshold: float = 0.85,
    ) -> None:
        self.ontology_path = ontology_path
        self.entity_dedup_threshold = entity_dedup_threshold
        self._ontology: dict[str, Any] | None = None

    @property
    def ontology(self) -> dict[str, Any]:
        if self._ontology is None and self.ontology_path:
            with open(self.ontology_path) as f:
                self._ontology = yaml.safe_load(f)
        return self._ontology or {}

    # ── BaseEvaluator contract ──

    def evaluate(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Run full structural audit and return a report dict."""
        report: dict[str, Any] = {
            "graph_stats": self._basic_stats(graph, triples),
            "degree_distribution": self._degree_distribution(graph),
            "connectivity_stats": self._connectivity_stats(graph),
            "orphan_analysis": self._orphan_analysis(graph, entities),
            "density_analysis": self._density_analysis(graph),
            "schema_compliance": self._schema_compliance(graph, triples),
            "constraint_audit": self._constraint_audit(entities, triples),
            "entity_duplication": self._entity_duplication(entities),
            "multi_hop_connectivity": self._multi_hop_connectivity(graph),
            "overall_health_score": 0.0,  # computed below
        }

        # Compute overall health score (0-100)
        scores = [
            report["orphan_analysis"]["health_score"],
            report["density_analysis"]["health_score"],
            report["schema_compliance"]["health_score"],
            report["constraint_audit"]["health_score"],
            report["entity_duplication"]["health_score"],
            report["multi_hop_connectivity"]["health_score"],
        ]
        report["overall_health_score"] = round(sum(scores) / len(scores), 1)

        # Add interpretation
        report["verdict"] = self._interpret(report["overall_health_score"])
        return report

    # ── Individual Audit Functions ────────────────────────────

    @staticmethod
    def _basic_stats(graph: nx.DiGraph, triples: list[tuple[str, str, str, str]]) -> dict[str, Any]:
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_triples": len(triples),
            "is_directed": graph.is_directed(),
            "is_connected": nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else False,
        }

    @staticmethod
    def _degree_distribution(graph: nx.DiGraph) -> dict[str, Any]:
        """Compute in-degree and out-degree distribution statistics.

        Returns summary stats (mean, median, min, max, stdev) plus a
        binned histogram for external visualization. Pure Python — no
        numpy dependency.
        """
        n = graph.number_of_nodes()
        if n == 0:
            return {
                "in_degree": {"mean": 0, "median": 0, "min": 0, "max": 0, "stdev": 0},
                "out_degree": {"mean": 0, "median": 0, "min": 0, "max": 0, "stdev": 0},
                "histogram": {"bin_edges": [], "in_counts": [], "out_counts": []},
            }

        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]

        def _summary(values: list[int]) -> dict[str, float]:
            return {
                "mean": round(statistics.mean(values), 2),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
            }

        # Build histogram bins (log-spaced for power-law distributions)
        all_vals = in_degrees + out_degrees
        max_deg = max(all_vals) if all_vals else 0
        if max_deg <= 1:
            bin_edges = [0, 1]
            in_counts = [in_degrees.count(0), in_degrees.count(1)]
            out_counts = [out_degrees.count(0), out_degrees.count(1)]
        else:
            # 10 log-spaced bins from 1 to max_deg, plus a [0] bin
            num_bins = min(10, max_deg)
            bin_edges = [0] + [
                round(math.exp(math.log(max_deg) * i / num_bins)) for i in range(1, num_bins + 1)
            ]
            # Deduplicate and ensure monotonic
            seen: set[int] = set()
            deduped: list[int] = []
            for b in bin_edges:
                if b not in seen:
                    seen.add(b)
                    deduped.append(b)
            bin_edges = deduped

            in_counts = []
            out_counts = []
            for i in range(len(bin_edges) - 1):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                in_counts.append(sum(1 for d in in_degrees if lo <= d < hi))
                out_counts.append(sum(1 for d in out_degrees if lo <= d < hi))
            # Last bin is inclusive
            in_counts.append(sum(1 for d in in_degrees if d >= bin_edges[-1]))
            out_counts.append(sum(1 for d in out_degrees if d >= bin_edges[-1]))

        return {
            "in_degree": _summary(in_degrees),
            "out_degree": _summary(out_degrees),
            "histogram": {
                "bin_edges": bin_edges,
                "in_counts": in_counts,
                "out_counts": out_counts,
            },
        }

    @staticmethod
    def _connectivity_stats(graph: nx.DiGraph) -> dict[str, Any]:
        """Additional connectivity metrics beyond basic stats.

        Computes strongly connected components, average clustering
        coefficient, and estimated diameter (via sampling for large graphs).
        """
        n = graph.number_of_nodes()
        if n == 0:
            return {
                "num_scc": 0,
                "largest_scc_size": 0,
                "scc_size_gt1": 0,
                "avg_clustering": 0.0,
                "estimated_diameter": 0,
            }

        # Strongly connected components
        sccs = list(nx.strongly_connected_components(graph))
        scc_sizes = sorted((len(c) for c in sccs), reverse=True)

        # Average clustering coefficient (on undirected view for directed graphs)
        try:
            avg_clustering = nx.average_clustering(graph)
        except Exception:
            avg_clustering = 0.0

        # Diameter estimate: sample up to 100 random nodes, compute max shortest
        # path length among reachable pairs. Exact diameter is O(n²) — too expensive.
        sample = random.sample(list(graph.nodes()), min(100, n))
        max_dist = 0
        reachable_pairs = 0
        for src in sample:
            lengths = nx.single_source_shortest_path_length(graph, src)
            if lengths:
                max_dist = max(max_dist, max(lengths.values()))
                reachable_pairs += len(lengths) - 1  # exclude self

        return {
            "num_scc": len(sccs),
            "largest_scc_size": scc_sizes[0] if scc_sizes else 0,
            "scc_size_gt1": sum(1 for s in scc_sizes if s > 1),
            "avg_clustering": round(avg_clustering, 6),
            "estimated_diameter": max_dist,
        }

    def _orphan_analysis(self, graph: nx.DiGraph, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Identify nodes with zero connections (orphans)."""
        n_nodes = graph.number_of_nodes()
        if n_nodes == 0:
            return {"orphan_count": 0, "orphan_rate": 0.0, "health_score": 100}

        orphans = list(nx.isolates(graph))
        orphan_count = len(orphans)
        orphan_rate = orphan_count / n_nodes

        # Filter to only entity nodes (not Chunk/Document) for actionable orphans
        entity_orphans = [
            n for n in orphans if graph.nodes[n].get("type") not in ("Chunk", "Document")
        ]
        chunk_orphans = len(orphans) - len(entity_orphans)

        # Health score: 100 = no orphans, 0 = all orphans (harsh above 30%)
        health = max(0, 100 - (orphan_rate * 300))

        return {
            "orphan_count": orphan_count,
            "orphan_rate": round(orphan_rate, 4),
            "entity_orphans": len(entity_orphans),
            "chunk_orphans": chunk_orphans,
            "health_score": round(health, 1),
            "flag": "red" if orphan_rate > 0.3 else ("yellow" if orphan_rate > 0.1 else "green"),
            "recommendation": (
                "High orphan rate — consider merging orphan entities or enriching connections."
                if orphan_rate > 0.3
                else ""
            ),
        }

    @staticmethod
    def _density_analysis(graph: nx.DiGraph) -> dict[str, Any]:
        """Graph density: too sparse = isolated facts; too dense = over-connected noise."""
        n = graph.number_of_nodes()
        if n < 2:
            return {"density": 0.0, "health_score": 100}

        density = nx.density(graph)

        # For a KG: ideal density is typically 0.01-0.05 (sparse but connected)
        # Too sparse (<0.005) = dead facts; too dense (>0.2) = possible noise
        if density < 0.005:
            health = max(0, density * 2000)  # scale up
        elif density <= 0.2:
            health = 100  # sweet spot
        else:
            health = max(0, 100 - (density - 0.2) * 500)

        return {
            "density": round(density, 6),
            "health_score": round(health, 1),
            "flag": "red" if density < 0.005 else ("yellow" if density > 0.2 else "green"),
            "recommendation": (
                "Graph is very sparse — relations may not capture enough context for multi-hop reasoning."
                if density < 0.005
                else (
                    "Graph is very dense — possible over-connected noise; verify relation quality."
                    if density > 0.2
                    else ""
                )
            ),
        }

    def _schema_compliance(
        self,
        graph: nx.DiGraph,
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Check if edges follow ontology schema rules (e.g., works_at: PERSON→ORG)."""
        if not self.ontology or not triples:
            return {
                "compliance_rate": 1.0,
                "violations": [],
                "health_score": 100,
                "flag": "green",
            }

        rel_rules = self.ontology.get("relationship_types", {})
        if not rel_rules:
            return {
                "compliance_rate": 1.0,
                "violations": [],
                "health_score": 100,
                "flag": "green",
            }

        violations: list[dict[str, str]] = []
        for subj, pred, obj, _ in triples:
            rule = rel_rules.get(pred, {})
            if not rule:
                continue  # No rule defined = no violation

            expected_domain = rule.get("domain", "")
            expected_range = rule.get("range", "")

            subj_type = graph.nodes[subj].get("type", "") if subj in graph.nodes else ""
            obj_type = graph.nodes[obj].get("type", "") if obj in graph.nodes else ""

            if expected_domain and subj_type and subj_type != expected_domain:
                violations.append(
                    {
                        "triple": f"({subj} -{pred}-> {obj})",
                        "issue": f"Subject type '{subj_type}' ≠ expected domain '{expected_domain}'",
                    }
                )
            if expected_range and obj_type and obj_type != expected_range:
                violations.append(
                    {
                        "triple": f"({subj} -{pred}-> {obj})",
                        "issue": f"Object type '{obj_type}' ≠ expected range '{expected_range}'",
                    }
                )

        compliance_rate = 1.0 - (len(violations) / len(triples))
        health = round(compliance_rate * 100, 1)

        return {
            "compliance_rate": round(compliance_rate, 4),
            "violation_count": len(violations),
            "violations": violations[:10],  # cap for readability
            "health_score": health,
            "flag": "red"
            if compliance_rate < 0.8
            else ("yellow" if compliance_rate < 0.95 else "green"),
            "recommendation": (
                f"{len(violations)} schema violations found — check ontology rules."
                if violations
                else ""
            ),
        }

    def _constraint_audit(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Check functional property violations on attributes.

        Reads the ontology's ``attributes`` section for any property marked
        ``functional: true`` (at most one value per entity). Flags entities
        that have multiple distinct values for the same functional attribute.

        Also detects contradictory facts: when two triples assert different
        values for the same entity's functional attribute.
        """
        if not self.ontology:
            return {
                "functional_violations": [],
                "violation_count": 0,
                "contradictory_facts": 0,
                "health_score": 100,
                "flag": "green",
                "recommendation": "",
            }

        # Collect functional attribute names by entity type
        attr_schema = self.ontology.get("attributes", {})
        functional_attrs: dict[str, set[str]] = {}  # entity_type -> {attr_name, ...}
        for etype, attrs in attr_schema.items():
            funcs: set[str] = set()
            if isinstance(attrs, dict):
                for attr_name, meta in attrs.items():
                    if isinstance(meta, dict) and meta.get("functional"):
                        funcs.add(attr_name)
            elif isinstance(attrs, list):
                # Old format: flat list, no functional markers
                pass
            if funcs:
                functional_attrs[etype] = funcs

        if not functional_attrs:
            return {
                "functional_violations": [],
                "violation_count": 0,
                "contradictory_facts": 0,
                "health_score": 100,
                "flag": "green",
                "recommendation": "",
            }

        violations: list[dict[str, Any]] = []

        # 1. Check entity dicts for list-valued functional attributes
        for e in entities:
            etype = e.get("type", "")
            funcs = functional_attrs.get(etype, set())
            if not funcs:
                continue
            for attr in funcs:
                val = e.get(attr)
                if isinstance(val, list) and len(val) > 1:
                    violations.append(
                        {
                            "entity": e.get("name", "?"),
                            "type": etype,
                            "attribute": attr,
                            "values": val,
                            "issue": f"Functional attribute '{attr}' has {len(val)} values",
                        }
                    )

        # 2. Check triples for contradictory facts (same subj+pred, different obj)
        # Build a map of (subject, predicate) -> set of objects
        triple_map: dict[tuple[str, str], set[str]] = defaultdict(set)
        for subj, pred, obj, _ in triples:
            triple_map[(subj, pred)].add(obj)

        contradictory_count = 0
        for (subj, pred), objects in triple_map.items():
            if len(objects) > 1:
                # Check if this predicate corresponds to a functional attribute
                # (attributes can appear as triples too, e.g., has_birth_date)
                for __, funcs in functional_attrs.items():
                    if pred in funcs or pred.startswith("has_") and pred[4:] in funcs:
                        contradictory_count += 1
                        violations.append(
                            {
                                "entity": subj,
                                "attribute": pred,
                                "values": sorted(objects),
                                "issue": f"Contradictory values for functional predicate '{pred}'",
                            }
                        )
                        break

        violation_count = len(violations)
        # Health: each violation costs 10 points, floor at 0
        health = max(0, 100 - violation_count * 10)
        violation_rate = violation_count / max(len(entities), 1)

        return {
            "functional_violations": violations[:20],
            "violation_count": violation_count,
            "contradictory_facts": contradictory_count,
            "violation_rate": round(violation_rate, 4),
            "health_score": round(health, 1),
            "flag": "red"
            if violation_rate > 0.1
            else ("yellow" if violation_count > 0 else "green"),
            "recommendation": (
                f"{violation_count} functional property violations — "
                f"{contradictory_count} contradictory facts found."
                if violations
                else ""
            ),
        }

    def _entity_duplication(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect potential duplicate entities using name similarity.

        Uses a lightweight character n-gram Jaccard approach (no embedding model needed).
        For production, replace with sentence-transformers cosine similarity.
        """
        if len(entities) < 2:
            return {
                "duplicate_pairs": [],
                "duplicate_entity_count": 0,
                "health_score": 100,
                "flag": "green",
            }

        names = [(e.get("name", ""), e.get("type", "")) for e in entities]
        duplicates: list[dict[str, Any]] = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = self._ngram_jaccard(names[i][0], names[j][0])
                if sim >= self.entity_dedup_threshold:
                    duplicates.append(
                        {
                            "entity_a": names[i][0],
                            "entity_b": names[j][0],
                            "similarity": round(sim, 4),
                        }
                    )

        dup_ratio = len(duplicates) / max(len(entities), 1)
        health = max(0, 100 - dup_ratio * 500)

        return {
            "duplicate_pairs": duplicates[:20],
            "all_duplicate_pairs": duplicates,
            "duplicate_pair_count": len(duplicates),
            "duplicate_entity_count": len(
                {d["entity_a"] for d in duplicates} | {d["entity_b"] for d in duplicates}
            ),
            "health_score": round(health, 1),
            "flag": "red"
            if len(duplicates) > len(entities) * 0.1
            else ("yellow" if duplicates else "green"),
            "recommendation": (
                f"{len(duplicates)} potential duplicate pairs — consider entity resolution."
                if duplicates
                else ""
            ),
        }

    @staticmethod
    def _multi_hop_connectivity(graph: nx.DiGraph) -> dict[str, Any]:
        """Measure how many nodes are reachable in 2-3 hops (needed for chain reasoning)."""
        n = graph.number_of_nodes()
        if n < 3:
            return {
                "reachable_2hop_pct": 0.0,
                "reachable_3hop_pct": 0.0,
                "avg_path_length": 0.0,
                "health_score": 100,
                "flag": "green",
            }

        # Sample to keep computation fast for large graphs
        sample_nodes = list(graph.nodes())[:200]
        hop2_reachable = 0
        hop3_reachable = 0
        total_pairs = 0

        for node in sample_nodes:
            # BFS limited to 2 and 3 hops
            visited_2 = set()
            visited_3 = set()
            queue = [(node, 0)]

            while queue:
                current, depth = queue.pop(0)
                if depth > 3:
                    break
                for neighbor in graph.successors(current):
                    if neighbor not in visited_2 and neighbor not in visited_3:
                        if depth + 1 <= 2:
                            visited_2.add(neighbor)
                        if depth + 1 <= 3:
                            visited_3.add(neighbor)
                        queue.append((neighbor, depth + 1))

            hop2_reachable += len(visited_2)
            hop3_reachable += len(visited_3)
            total_pairs += len(sample_nodes) - 1  # rough upper bound

        pct_2hop = hop2_reachable / max(total_pairs, 1)
        pct_3hop = hop3_reachable / max(total_pairs, 1)

        # Health: more reachable nodes in 2-3 hops = better for multi-hop QA
        avg_reachability = (pct_2hop + pct_3hop) / 2
        health = min(100, avg_reachability * 200)

        return {
            "reachable_2hop_pct": round(pct_2hop, 4),
            "reachable_3hop_pct": round(pct_3hop, 4),
            "health_score": round(health, 1),
            "flag": "red"
            if avg_reachability < 0.1
            else ("yellow" if avg_reachability < 0.3 else "green"),
            "recommendation": (
                "Low multi-hop connectivity — the KG may not support chain reasoning questions."
                if avg_reachability < 0.1
                else ""
            ),
        }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
        """Character n-gram Jaccard similarity for lightweight dedup detection."""
        if not a or not b:
            return 0.0
        a = a.lower().strip()
        b = b.lower().strip()
        if a == b:
            return 1.0
        ngrams_a = {a[i : i + n] for i in range(len(a) - n + 1)}
        ngrams_b = {b[i : i + n] for i in range(len(b) - n + 1)}
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = ngrams_a & ngrams_b
        union = ngrams_a | ngrams_b
        return len(intersection) / len(union)

    @staticmethod
    def _interpret(score: float) -> str:
        if score >= 80:
            return "✅ Healthy — KG is structurally sound for SFT data generation."
        elif score >= 60:
            return "⚠️ Fair — some issues detected; review flagged areas before generating SFT data."
        elif score >= 40:
            return "🔶 Concerning — multiple structural issues; SFT data quality may suffer."
        else:
            return "🔴 Poor — significant structural problems; fix before using for LLM training."


def load_kg_for_audit(
    path: Path,
) -> tuple[nx.DiGraph, list[dict[str, Any]], list[tuple[str, str, str, str]]]:
    """Load a KG from a JSON export file for auditing."""
    with open(path) as f:
        data = json.load(f)

    entities = data.get("entities", [])
    triples_raw = data.get("triples", [])
    graph = nx.node_link_graph(data.get("graph", {}), link="edges")

    triples: list[tuple[str, str, str, str]] = []
    for t in triples_raw:
        if isinstance(t, list | tuple):
            if len(t) >= 4:
                triples.append((str(t[0]), str(t[1]), str(t[2]), str(t[3])))
            elif len(t) == 3:
                triples.append((str(t[0]), str(t[1]), str(t[2]), ""))

    return graph, entities, triples
