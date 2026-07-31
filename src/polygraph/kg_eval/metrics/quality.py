"""Quality evaluation metrics for knowledge graphs."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from polygraph.kg_eval._base import BaseEvaluator

logger = logging.getLogger(__name__)


class QualityEvaluator(BaseEvaluator):
    """
    Evaluates KG quality against metrics from the Problem Description:
    completeness, consistency, duplication level, missing information,
    format errors, labeling quality, and reusability.
    """

    def __init__(self, ontology_path: Path | None = None) -> None:
        """Initialize with optional ontology for schema-aware metrics.

        Args:
            ontology_path: Path to an ontology YAML file (e.g.,
                ``configs/default_ontology.yaml``). When provided, enables
                schema completeness and attribute completeness checks.
        """
        self.ontology_path = ontology_path
        self._ontology: dict[str, Any] | None = None

    @property
    def ontology(self) -> dict[str, Any]:
        if self._ontology is None and self.ontology_path:
            with open(self.ontology_path) as f:
                self._ontology = yaml.safe_load(f)
        return self._ontology or {}

    # ── File I/O convenience (not part of BaseEvaluator contract) ──

    @staticmethod
    def evaluate_file(path: Path) -> dict[str, Any]:
        """Load a serialized KG from a file path and evaluate it."""
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            entities = data.get("entities", [])
            triples = data.get("triples", [])
            graph = nx.node_link_graph(data.get("graph", {}))
        else:
            logger.warning(f"Unsupported format: {path.suffix}")
            return {}

        return QualityEvaluator().evaluate(graph, entities, triples)

    # ── BaseEvaluator contract ──

    def evaluate(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Compute all quality metrics for a graph."""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_triples": len(triples),
            **self.completeness(entities),
            **self.consistency(graph, triples),
            **self.duplication_level(entities, triples),
            **self.missing_information(entities),
            **self.format_errors(triples),
            **self.labeling_quality(entities, graph),
            **self.reusability_score(graph, entities, triples),
            **self.schema_completeness(entities, triples),
            **self.syntactic_accuracy(entities, triples),
            "overall_score": self._overall_score(graph, entities, triples),
        }

    # ── Individual Metrics ──

    def completeness(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Fraction of entities that have all key fields populated.

        When an ontology is loaded via the constructor, checks fill rates
        for all ontology-defined attributes per entity type (not just
        name/type/aliases). Otherwise falls back to the default three-field
        check.
        """
        if not entities:
            return {"completeness": 0.0, "completeness_breakdown": {}}

        # Determine which fields to check per entity type
        ontology_types = self.ontology.get("entity_types", {}) if self.ontology else {}

        if ontology_types:
            # Ontology-aware: check all defined attributes per entity type
            all_scores: list[float] = []
            breakdown: dict[str, float] = {}

            # Group entities by type
            by_type: dict[str, list[dict[str, Any]]] = {}
            for e in entities:
                etype = e.get("type", "__unknown__")
                by_type.setdefault(etype, []).append(e)

            for etype, group in by_type.items():
                attrs = ontology_types.get(etype, {}).get("attributes", {})
                if not attrs:
                    # No ontology info for this type — check name/type/aliases only
                    for field in ("name", "type", "aliases"):
                        filled = sum(1 for e in group if e.get(field))
                        score = filled / len(group)
                        breakdown[f"{etype}.has_{field}"] = score
                        all_scores.append(score)
                else:
                    for attr_name in attrs:
                        filled = sum(1 for e in group if e.get(attr_name))
                        score = filled / len(group)
                        breakdown[f"{etype}.{attr_name}"] = score
                        all_scores.append(score)

            return {
                "completeness": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
                "completeness_breakdown": breakdown,
            }

        # Fallback: default three-field check (no ontology loaded)
        expected_fields = {"name", "type", "aliases"}
        scores = []
        breakdown = {}

        for field in expected_fields:
            filled = sum(1 for e in entities if e.get(field))
            score = filled / len(entities)
            breakdown[f"has_{field}"] = score
            scores.append(score)

        return {
            "completeness": sum(scores) / len(scores),
            "completeness_breakdown": breakdown,
        }

    def consistency(
        self,
        graph: nx.DiGraph,
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, float]:
        """Schema conformance and structural consistency."""
        if not triples:
            return {"consistency": 1.0}

        node_names = set(graph.nodes())
        orphan_count = sum(1 for t in triples if t[0] not in node_names or t[2] not in node_names)
        endpoint_score = 1.0 - (orphan_count / len(triples))

        label_conflicts = 0
        for node in graph.nodes():
            data = graph.nodes[node]
            if data.get("type") in ("UNKNOWN", "", None):
                label_conflicts += 1
        label_score = 1.0 - (label_conflicts / max(graph.number_of_nodes(), 1))

        score = (endpoint_score + label_score) / 2
        return {"consistency": score}

    def duplication_level(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, float]:
        """Detect duplicate entities and triples."""
        if not entities or not triples:
            return {"duplication_level": 0.0}

        # Entity name duplication
        names = [e["name"].lower() for e in entities]
        entity_dup_ratio = 1.0 - (len(set(names)) / len(names))

        # Triple duplication (ignore source_text for dedup comparison)
        triple_keys = {(t[0], t[1], t[2]) for t in triples}
        triple_dup_ratio = 1.0 - (len(triple_keys) / len(triples))

        score = (entity_dup_ratio + triple_dup_ratio) / 2
        return {
            "duplication_level": score,
            "entity_duplication_rate": entity_dup_ratio,
            "triple_duplication_rate": triple_dup_ratio,
        }

    def missing_information(self, entities: list[dict[str, Any]]) -> dict[str, float]:
        """Fraction of entities with empty/missing type or aliases."""
        if not entities:
            return {"missing_information": 1.0}

        missing_count = sum(
            1
            for e in entities
            if not e.get("type") or not e.get("aliases") or e.get("type") in ("UNKNOWN", "")
        )
        return {"missing_information": missing_count / len(entities)}

    def format_errors(self, triples: list[tuple[str, str, str, str]]) -> dict[str, float]:
        """Detect malformed triples (empty strings, wrong types)."""
        if not triples:
            return {"format_errors": 0.0}

        errors = 0
        for t in triples:
            if (
                not t[0]
                or not t[1]
                or not t[2]
                or not isinstance(t[0], str)
                or not isinstance(t[1], str)
                or not isinstance(t[2], str)
            ):
                errors += 1

        return {"format_errors": errors / len(triples)}

    def labeling_quality(
        self,
        entities: list[dict[str, Any]],
        graph: nx.DiGraph,
    ) -> dict[str, float]:
        """Heuristic labeling quality — fraction of entities with meaningful types."""
        if not entities:
            return {"labeling_quality": 0.0}

        generic = {"ENTITY", "UNKNOWN", "NAMED_ENTITY", "CONCEPT"}
        meaningful = sum(1 for e in entities if e.get("type") and e["type"] not in generic)
        return {"labeling_quality": meaningful / len(entities)}

    def reusability_score(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, float]:
        """Score indicating how reusable the KG is for downstream tasks."""
        score = 0.0

        if graph.number_of_nodes() > 0:
            score += 0.2
        if graph.number_of_edges() > 0:
            score += 0.2

        if graph.number_of_edges() > 0:
            try:
                connected_ratio = len(max(nx.weakly_connected_components(graph), key=len)) / max(
                    graph.number_of_nodes(), 1
                )
            except Exception:
                connected_ratio = 0.5  # assume reasonable connectivity
            score += 0.2 * connected_ratio

        # Has meaningful types (not generic fallbacks)
        generic = {"ENTITY", "UNKNOWN", "Chunk", "Document"}
        if entities:
            meaningful = sum(1 for e in entities if e.get("type") not in generic)
            score += 0.2 * (meaningful / len(entities))

        # Has descriptions
        if entities:
            has_desc = sum(1 for e in entities if e.get("description"))
            score += 0.2 * (has_desc / len(entities))

        return {"reusability": score}

    # ── Schema & Syntactic Metrics (Steps 3, 4, 5) ──────────────

    def schema_completeness(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Measure how many ontology-defined types and relations appear in the KG.

        Requires an ontology YAML loaded via the constructor. Returns coverage
        ratios for both entity types and relationship types. When no ontology
        is loaded, returns empty results.
        """
        if not self.ontology:
            return {}

        # Entity type coverage
        defined_entity_types = set(self.ontology.get("entity_types", {}).keys())
        present_entity_types = {e.get("type", "") for e in entities if e.get("type")}
        present_entity_types.discard("")
        missing_entity_types = defined_entity_types - present_entity_types
        entity_coverage = (
            len(present_entity_types & defined_entity_types) / len(defined_entity_types)
            if defined_entity_types
            else 1.0
        )

        # Relationship type coverage
        defined_rel_types = set(self.ontology.get("relationship_types", {}).keys())
        present_rel_types = {t[1] for t in triples if t[1]}
        missing_rel_types = defined_rel_types - present_rel_types
        rel_coverage = (
            len(present_rel_types & defined_rel_types) / len(defined_rel_types)
            if defined_rel_types
            else 1.0
        )

        return {
            "schema_entity_coverage": round(entity_coverage, 4),
            "schema_relation_coverage": round(rel_coverage, 4),
            "schema_entity_types_present": sorted(present_entity_types & defined_entity_types),
            "schema_entity_types_missing": sorted(missing_entity_types),
            "schema_relation_types_present": sorted(present_rel_types & defined_rel_types),
            "schema_relation_types_missing": sorted(missing_rel_types),
        }

    # ── Syntactic Accuracy ─────────────────────────────────────

    # Pattern for valid IDs: alphanumeric + hyphens, underscores, periods
    _VALID_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

    # Common ISO 8601 date formats
    _ISO_DATE_PATTERNS = [
        re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # 2024-01-15
        re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"),  # 2024-01-15T10:30
        re.compile(r"^\d{4}$"),  # 2024
    ]

    @staticmethod
    def _looks_like_date(value: str) -> bool:
        """Check if a string value looks like a date."""
        return any(pat.match(value.strip()) for pat in QualityEvaluator._ISO_DATE_PATTERNS)

    @staticmethod
    def _looks_like_number(value: str) -> bool:
        """Check if a string value looks numeric."""
        stripped = value.strip()
        try:
            float(stripped)
            return True
        except ValueError:
            return False

    def syntactic_accuracy(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> dict[str, Any]:
        """Extended format validation beyond empty-string checks.

        Checks:
          - ID format: entity names should match alphanumeric + [-_.] pattern
          - Date values: string values that look like dates should be valid ISO 8601
          - Numeric values: string values that look numeric should parse correctly
          - Predicate naming: relation predicates should follow snake_case convention
        """
        len(entities)
        len(triples)
        total_checks = 0
        errors: dict[str, int] = {
            "bad_id_format": 0,
            "bad_date_format": 0,
            "bad_numeric_format": 0,
            "bad_predicate_naming": 0,
        }
        examples: dict[str, list[str]] = {k: [] for k in errors}

        # 1. ID format check on entity names
        for e in entities:
            total_checks += 1
            name = e.get("name", "")
            if name and not self._VALID_ID_PATTERN.match(str(name)):
                errors["bad_id_format"] += 1
                if len(examples["bad_id_format"]) < 5:
                    examples["bad_id_format"].append(str(name))

        # 2 & 3. Date and numeric checks on entity attribute values
        for e in entities:
            for key, val in e.items():
                if key in ("name", "type", "aliases", "description", "sourceDocument"):
                    continue
                if not isinstance(val, str) or not val.strip():
                    continue
                total_checks += 1
                if self._looks_like_date(val):
                    # Verify it's a valid ISO 8601 date
                    try:
                        # Quick parse check — accept common formats
                        datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        errors["bad_date_format"] += 1
                        if len(examples["bad_date_format"]) < 5:
                            examples["bad_date_format"].append(f"{e.get('name', '?')}.{key}={val}")
                elif self._looks_like_number(val):
                    # Already verified by _looks_like_number — no further check needed
                    pass

        # 4. Predicate naming convention (snake_case expected)
        for triple in triples:
            pred = triple[1] if len(triple) > 1 else ""
            total_checks += 1
            if pred and not re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", pred):
                errors["bad_predicate_naming"] += 1
                if len(examples["bad_predicate_naming"]) < 5:
                    examples["bad_predicate_naming"].append(pred)

        total_errors = sum(errors.values())
        error_rate = total_errors / max(total_checks, 1)

        return {
            "syntactic_accuracy": round(1.0 - error_rate, 4),
            "syntactic_error_rate": round(error_rate, 4),
            "syntactic_error_breakdown": errors,
            "syntactic_error_examples": {k: v for k, v in examples.items() if v},
        }

    def _overall_score(
        self,
        graph: nx.DiGraph,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, str, str, str]],
    ) -> float:
        # Compute each metric individually (not via evaluate_graph to avoid recursion)
        comp = self.completeness(entities).get("completeness", 0.0)
        cons = self.consistency(graph, triples).get("consistency", 0.0)
        dup = self.duplication_level(entities, triples).get("duplication_level", 0.0)
        missing = self.missing_information(entities).get("missing_information", 0.0)
        fmt_err = self.format_errors(triples).get("format_errors", 0.0)
        label_q = self.labeling_quality(entities, graph).get("labeling_quality", 0.0)
        reuse = self.reusability_score(graph, entities, triples).get("reusability", 0.0)

        scores = [comp, cons, label_q, reuse, 1.0 - dup, 1.0 - fmt_err, 1.0 - missing]
        return sum(scores) / len(scores)
