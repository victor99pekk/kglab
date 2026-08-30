"""NetworkX graph-construction method."""

import logging
from typing import Any

import networkx as nx

from kglab._shared import GraphBackend, Ontology, entity_id

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a directed knowledge graph from resolved entities and triples."""

    def __init__(
        self,
        ontology: Ontology | None = None,
        backend: GraphBackend = GraphBackend.NETWORKX,
    ) -> None:
        self.ontology = ontology
        self.backend = backend

    def build(
        self,
        entities: list[dict[str, Any]],
        triples: list[tuple[str, ...]],
    ) -> nx.DiGraph:
        """Build a graph from ID-based triples with evidence and provenance."""
        graph = nx.DiGraph()
        entity_types: dict[str, str] = {}

        for entity in entities:
            node_type = entity.get("type", entity.get("label", "ENTITY"))
            node_id = entity.get("id") or entity_id(node_type, entity.get("name", ""))
            entity_types[node_id] = node_type
            node_data = {
                "id": node_id,
                "name": entity.get("name", ""),
                "type": node_type,
                "aliases": entity.get("aliases", []),
                "description": entity.get("description", ""),
                "importanceScore": entity.get("importanceScore", 0.0),
                "confidenceScore": entity.get("confidenceScore", 1.0),
                "source": entity.get("source", []),
                "source_chunk_ids": entity.get("source_chunk_ids", []),
                "embedding": entity.get("embedding"),
                "updatedAt": entity.get("updatedAt", ""),
                "text": entity.get("text", ""),
                "tokenCount": entity.get("tokenCount", 0),
                "index": entity.get("index", 0),
                "chunk_count": entity.get("chunk_count", 0),
                "upload_date": entity.get("upload_date", ""),
            }
            for key in (
                "title",
                "url",
                "license",
                "source_domain",
                "scraped_at",
                "crawler",
                "content_hash",
                "inferred_type",
            ):
                if entity.get(key) not in (None, ""):
                    node_data[key] = entity[key]
            graph.add_node(node_id, **node_data)

        if self.ontology:
            self._validate_structural_triples(triples, entity_types)

        for triple in triples:
            subject, predicate, object_id = triple[0], triple[1], triple[2]
            evidence_sentence = triple[3] if len(triple) > 3 else ""
            source_chunk_id = triple[4] if len(triple) > 4 else ""
            relation_record = {
                "predicate": predicate,
                "evidence_sentence": evidence_sentence,
                "source_chunk_id": source_chunk_id,
            }
            if len(triple) > 5:
                relation_record["description"] = triple[5]
            if len(triple) > 6:
                relation_record["confidenceScore"] = triple[6]

            if subject not in graph:
                graph.add_node(subject, id=subject, type="ENTITY", name=subject)
            if object_id not in graph:
                graph.add_node(object_id, id=object_id, type="ENTITY", name=object_id)

            if graph.has_edge(subject, object_id):
                edge = graph.edges[subject, object_id]
                predicates = edge.get("predicates", [])
                if predicate not in predicates:
                    predicates.append(predicate)
                edge["predicates"] = predicates
                edge["weight"] = len(predicates)

                source_texts = edge.get("source_texts", [])
                if evidence_sentence and evidence_sentence not in source_texts:
                    source_texts.append(evidence_sentence)
                    edge["source_texts"] = source_texts

                source_chunks = edge.get("source_chunk_ids", [])
                if source_chunk_id and source_chunk_id not in source_chunks:
                    source_chunks.append(source_chunk_id)
                    edge["source_chunk_ids"] = source_chunks

                relations = edge.get("relations", [])
                if relation_record not in relations:
                    relations.append(relation_record)
                    edge["relations"] = relations
                if relation_record.get("description"):
                    edge["description"] = relation_record["description"]
            else:
                graph.add_edge(
                    subject,
                    object_id,
                    predicates=[predicate],
                    weight=1,
                    source_texts=[evidence_sentence] if evidence_sentence else [],
                    source_chunk_ids=[source_chunk_id] if source_chunk_id else [],
                    relations=[relation_record],
                    description=relation_record.get("description", ""),
                )

        self._compute_importance(graph)
        logger.info(
            "Built graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        if self.ontology:
            self._validate(graph)
        return graph

    def _validate_structural_triples(
        self,
        triples: list[tuple[str, ...]],
        entity_types: dict[str, str],
    ) -> None:
        """Reject structural edges whose endpoints violate the ontology."""
        if not self.ontology:
            return

        constraints = self.ontology.get_structural_relation_types()
        violations: list[str] = []
        for triple in triples:
            subject, predicate, object_id = triple[0], triple[1], triple[2]
            if predicate not in constraints:
                continue

            expected_subject, expected_object = constraints[predicate]
            subject_type = entity_types.get(subject)
            object_type = entity_types.get(object_id)
            if subject_type is None or object_type is None:
                missing = [
                    node_id
                    for node_id, node_type in ((subject, subject_type), (object_id, object_type))
                    if node_type is None
                ]
                violations.append(f"{predicate}: missing endpoint(s) {', '.join(missing)}")
                continue

            subject_matches = (
                expected_subject == "Entity" and subject_type not in {"Chunk", "Document"}
            ) or subject_type == expected_subject
            object_matches = object_type == expected_object
            if not subject_matches or not object_matches:
                violations.append(
                    f"{predicate}: expected {expected_subject}->{expected_object}, "
                    f"got {subject_type}->{object_type} ({subject}->{object_id})"
                )

        if violations:
            preview = "; ".join(violations[:5])
            remainder = len(violations) - 5
            if remainder > 0:
                preview += f"; and {remainder} more"
            raise ValueError(f"Invalid structural relationship(s): {preview}")

    @staticmethod
    def _compute_importance(graph: nx.DiGraph) -> None:
        """Store PageRank importance on each node when it can be computed."""
        try:
            scores = nx.pagerank(graph, max_iter=100, tol=1e-4)
            for node, score in scores.items():
                graph.nodes[node]["importanceScore"] = round(score, 6)
        except Exception:
            logger.debug("PageRank computation failed; skipping importance scores")

    def _validate(self, graph: nx.DiGraph) -> None:
        """Log graph elements that are absent from the configured ontology."""
        if not self.ontology:
            return

        # The KG always contains three structural node kinds — entities,
        # documents, and chunks — regardless of the ontology.  Documents and
        # chunks are not *entity* types, so they are always accepted here.
        structural_kinds = {"document", "chunk"}
        ontology_labels = {
            label.casefold() for label in self.ontology.entity_types
        } | structural_kinds
        ontology_relations = set(self.ontology.relationship_types)
        node_mismatches = sum(
            1
            for _, data in graph.nodes(data=True)
            if (label := data.get("type", data.get("label", "")))
            and label.casefold() not in ontology_labels
        )
        relation_mismatches = sum(
            predicate not in ontology_relations
            for _, _, data in graph.edges(data=True)
            for predicate in data.get("predicates", [])
        )
        if node_mismatches:
            logger.warning(
                "Ontology validation: %d node labels not in schema",
                node_mismatches,
            )
        if relation_mismatches:
            logger.warning(
                "Ontology validation: %d relation types not in schema",
                relation_mismatches,
            )

    def stats(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Return summary graph statistics."""
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "num_connected_components": nx.number_weakly_connected_components(graph),
            "avg_degree": sum(dict(graph.degree()).values()) / max(graph.number_of_nodes(), 1),
            "label_distribution": self._label_distribution(graph),
            "relation_distribution": self._relation_distribution(graph),
        }

    @staticmethod
    def _label_distribution(graph: nx.DiGraph) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for _, data in graph.nodes(data=True):
            label = data.get("type", data.get("label", "UNKNOWN"))
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    @staticmethod
    def _relation_distribution(graph: nx.DiGraph) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            for predicate in data.get("predicates", []):
                distribution[predicate] = distribution.get(predicate, 0) + 1
        return distribution
