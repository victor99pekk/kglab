#!/usr/bin/env python3
"""Build KG1-A.2: a provenance-preserving Wikipedia/spaCy evidence graph."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = SCRIPT_DIR / "fixtures" / "pilot_articles.jsonl"
DEFAULT_TAXONOMY = EXPERIMENT_DIR / "wikimedia_topics_64.yaml"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "artifacts" / "kg1_spacy_pilot"

QID_RE = re.compile(r"^Q[1-9][0-9]*$")
GNN_NODE_TYPES = {"article", "chunk", "entity"}
GNN_EDGE_TYPES = {"describes", "has_chunk", "links_to", "mentions", "next"}

NODE_COLORS = {
    "article": "#2563eb",
    "chunk": "#94a3b8",
    "mention": "#f59e0b",
    "entity": "#10b981",
}


def _hash(*parts: object, length: int = 20) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _same_surface(first: str, second: str) -> bool:
    def without_determiner(value: str) -> str:
        normalized = _normalized_name(value)
        for prefix in ("a ", "an ", "the "):
            if normalized.startswith(prefix):
                return normalized[len(prefix) :]
        return normalized

    return without_determiner(first) == without_determiner(second)


def _article_id(record: dict[str, Any]) -> str:
    qid = str(record.get("qid", ""))
    if QID_RE.fullmatch(qid):
        return f"article:wikidata:{qid}"
    return f"article:wikipedia:en:{record['page_id']}"


def _linked_article_id(link: dict[str, Any]) -> str:
    qid = str(link.get("qid", ""))
    if QID_RE.fullmatch(qid):
        return f"article:wikidata:{qid}"
    return f"article:wikipedia:en:title:{_hash(link['title'])}"


def _entity_id(name: str, label: str, qid: str = "") -> str:
    if QID_RE.fullmatch(qid):
        return f"entity:wikidata:{qid}"
    return f"entity:local:{label.casefold()}:{_hash(label, _normalized_name(name))}"


def _chunk_id(article_id: str, start: int, end: int) -> str:
    return f"chunk:{_hash(article_id, start, end)}"


def _mention_id(article_id: str, start: int, end: int, extractor: str) -> str:
    return f"mention:{_hash(article_id, start, end, extractor)}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"article_id", "page_id", "title", "article_revision_id", "text"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            records.append(record)
    return records


class EvidenceGraph:
    """Deduplicated node/edge store with JSON-friendly records."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self._edge_keys: set[tuple[str, str, str, str]] = set()

    def add_node(self, node_id: str, node_type: str, **attributes: Any) -> None:
        incoming = {"id": node_id, "type": node_type, **attributes}
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = incoming
            return

        for key, value in incoming.items():
            if value not in (None, "", []):
                existing[key] = value

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        evidence_id: str = "",
        **attributes: Any,
    ) -> None:
        key = (source, relation, target, evidence_id)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(
            {
                "source": source,
                "relation": relation,
                "target": target,
                **({"evidence_id": evidence_id} if evidence_id else {}),
                **attributes,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "kg1-a.2",
            "nodes": sorted(self.nodes.values(), key=lambda node: node["id"]),
            "edges": sorted(
                self.edges,
                key=lambda edge: (
                    edge["source"],
                    edge["relation"],
                    edge["target"],
                    edge.get("evidence_id", ""),
                ),
            ),
        }


def _chunk_for_span(chunk_spans: list[tuple[int, int, str]], start: int, end: int) -> str:
    for chunk_start, chunk_end, chunk_id in chunk_spans:
        if chunk_start <= start and end <= chunk_end:
            return chunk_id
    raise ValueError(f"Mention span {start}:{end} does not belong to a chunk")


def _occurrences(text: str, surface: str) -> Iterable[tuple[int, int]]:
    if not surface:
        return
    pattern = re.compile(re.escape(surface), flags=re.IGNORECASE)
    for match in pattern.finditer(text):
        yield match.start(), match.end()


def _add_mention(
    graph: EvidenceGraph,
    *,
    article_id: str,
    chunk_id: str,
    entity_id: str,
    text: str,
    start: int,
    end: int,
    extractor: str,
    confidence: float | None,
    spacy_type: str = "",
    spacy_model: str = "",
) -> str:
    mention_id = _mention_id(article_id, start, end, extractor)
    graph.add_node(
        mention_id,
        "mention",
        text=text[start:end],
        start_char=start,
        end_char=end,
        chunk_id=chunk_id,
        extractor=extractor,
        confidence=confidence,
        spacy_type=spacy_type,
        spacy_model=spacy_model,
    )
    graph.add_edge(chunk_id, "has_mention", mention_id)
    graph.add_edge(mention_id, "refers_to", entity_id, evidence_id=chunk_id)
    graph.add_edge(
        chunk_id,
        "mentions",
        entity_id,
        evidence_id=mention_id,
        source_chunk_id=chunk_id,
    )
    graph.add_edge(
        article_id,
        "mentions",
        entity_id,
        evidence_id=chunk_id,
        source_chunk_id=chunk_id,
    )
    return mention_id


def _record_spacy_type(graph: EvidenceGraph, entity_id: str, label: str) -> None:
    """Aggregate mention-level spaCy predictions into entity feature evidence."""
    entity = graph.nodes[entity_id]
    counts = dict(entity.get("spacy_type_counts", {}))
    counts[label] = counts.get(label, 0) + 1
    entity["spacy_type_counts"] = dict(sorted(counts.items()))
    entity["spacy_types"] = sorted(counts)
    entity["primary_spacy_type"] = min(
        counts,
        key=lambda candidate: (-counts[candidate], candidate),
    )


def extract_article(record: dict[str, Any], nlp: Any, graph: EvidenceGraph) -> dict[str, Any]:
    """Add one article and its evidence; return separate supervision target."""
    text = str(record["text"])
    article_id = _article_id(record)
    revision_id = int(record["article_revision_id"])
    graph.add_node(
        article_id,
        "article",
        dataset_article_id=record["article_id"],
        page_id=int(record["page_id"]),
        qid=record.get("qid", ""),
        title=record["title"],
        article_revision_id=revision_id,
        split=record.get("split", ""),
        external_context=False,
    )
    subject_entity_id = ""
    subject_qid = str(record.get("qid", ""))
    if QID_RE.fullmatch(subject_qid):
        subject_entity_id = _entity_id(record["title"], "WIKIPEDIA_ENTITY", subject_qid)
        graph.add_node(
            subject_entity_id,
            "entity",
            name=record["title"],
            qid=subject_qid,
            canonical_source="wikipedia_page",
        )
        graph.add_edge(article_id, "describes", subject_entity_id)

    doc = nlp(text)
    chunk_spans: list[tuple[int, int, str]] = []
    previous_chunk_id = ""
    for chunk_index, sentence in enumerate(doc.sents):
        chunk_id = _chunk_id(article_id, sentence.start_char, sentence.end_char)
        chunk_spans.append((sentence.start_char, sentence.end_char, chunk_id))
        graph.add_node(
            chunk_id,
            "chunk",
            text=sentence.text,
            start_char=sentence.start_char,
            end_char=sentence.end_char,
            chunk_index=chunk_index,
            chunk_method="sentence",
            article_id=article_id,
            article_revision_id=revision_id,
        )
        graph.add_edge(article_id, "has_chunk", chunk_id)
        if previous_chunk_id:
            graph.add_edge(previous_chunk_id, "next", chunk_id)
        previous_chunk_id = chunk_id

    linked_spans: list[tuple[int, int, str, str]] = []
    link_occurrence_indexes: Counter[str] = Counter()
    for link in record.get("links", []):
        qid = str(link.get("qid", ""))
        entity_id = _entity_id(link["title"], "WIKIPEDIA_ENTITY", qid=qid)
        target_article_id = _linked_article_id(link)
        graph.add_node(
            entity_id,
            "entity",
            name=link["title"],
            qid=qid,
            canonical_source="wikipedia_link",
        )
        graph.add_node(
            target_article_id,
            "article",
            title=link["title"],
            qid=qid,
            external_context=True,
        )
        graph.add_edge(article_id, "links_to", target_article_id)

        surface = str(link["surface"])
        spans = list(_occurrences(text, surface))
        surface_key = _normalized_name(surface)
        occurrence_index = int(
            link.get("occurrence_index", link_occurrence_indexes[surface_key])
        )
        link_occurrence_indexes[surface_key] = occurrence_index + 1
        if occurrence_index >= len(spans):
            raise ValueError(
                f"Link occurrence {occurrence_index} for surface {surface!r} "
                f"missing from article {record['title']!r}"
            )
        start, end = spans[occurrence_index]
        chunk_id = _chunk_for_span(chunk_spans, start, end)
        mention_id = _add_mention(
            graph,
            article_id=article_id,
            chunk_id=chunk_id,
            entity_id=entity_id,
            text=text,
            start=start,
            end=end,
            extractor="wikipedia_link",
            confidence=1.0,
        )
        linked_spans.append((start, end, entity_id, mention_id))

    for entity in doc.ents:
        matching_links = [
            (entity_id, mention_id)
            for start, end, entity_id, mention_id in linked_spans
            if start < entity.end_char
            and entity.start_char < end
            and _same_surface(text[start:end], entity.text)
        ]
        subject_match = bool(
            subject_entity_id and _same_surface(entity.text, record["title"])
        )
        if matching_links:
            entity_id = matching_links[0][0]
        elif subject_match:
            entity_id = subject_entity_id
        else:
            entity_id = _entity_id(entity.text, entity.label_)
        graph.add_node(
            entity_id,
            "entity",
            name=(
                graph.nodes[entity_id]["name"]
                if matching_links or subject_match
                else entity.text
            ),
            canonical_source=(
                graph.nodes[entity_id].get("canonical_source", "spacy_ner")
                if matching_links or subject_match
                else "spacy_ner"
            ),
        )
        _record_spacy_type(graph, entity_id, entity.label_)
        spacy_model = nlp.meta.get("name", "unknown")

        if matching_links:
            for _, mention_id in matching_links:
                graph.add_node(
                    mention_id,
                    "mention",
                    spacy_type=entity.label_,
                    spacy_model=spacy_model,
                )
        else:
            chunk_id = _chunk_for_span(
                chunk_spans, entity.start_char, entity.end_char
            )
            _add_mention(
                graph,
                article_id=article_id,
                chunk_id=chunk_id,
                entity_id=entity_id,
                text=text,
                start=entity.start_char,
                end=entity.end_char,
                extractor="spacy_ner",
                confidence=None,
                spacy_type=entity.label_,
                spacy_model=spacy_model,
            )

    return {
        "article_id": article_id,
        "dataset_article_id": record["article_id"],
        "split": record.get("split", ""),
        "topic_ids": list(record.get("topic_ids", [])),
    }


def project_for_gnn(evidence: dict[str, Any]) -> dict[str, Any]:
    """Remove mention nodes while retaining chunks and entity type features."""
    nodes = [node for node in evidence["nodes"] if node["type"] in GNN_NODE_TYPES]
    node_ids = {node["id"] for node in nodes}
    edges = [
        edge
        for edge in evidence["edges"]
        if edge["relation"] in GNN_EDGE_TYPES
        and edge["source"] in node_ids
        and edge["target"] in node_ids
    ]
    return {
        "schema_version": evidence["schema_version"],
        "type_representation": "entity_attributes",
        "labels_as_message_edges": False,
        "nodes": nodes,
        "edges": edges,
    }


def validate_evidence_graph(evidence: dict[str, Any]) -> None:
    """Enforce the one-mention-to-one-entity resolution contract."""
    mention_ids = {
        node["id"] for node in evidence["nodes"] if node["type"] == "mention"
    }
    refers_to_counts = Counter(
        edge["source"]
        for edge in evidence["edges"]
        if edge["relation"] == "refers_to"
    )
    invalid = {
        mention_id: refers_to_counts[mention_id]
        for mention_id in mention_ids
        if refers_to_counts[mention_id] != 1
    }
    if invalid:
        raise ValueError(
            "Every mention must resolve to exactly one entity; "
            f"invalid mentions: {dict(sorted(invalid.items()))}"
        )


def build_label_graph(taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Build label hierarchy without article-label edges."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str, str]] = set()

    for domain in taxonomy["domains"]:
        domain_name = domain["id"]
        domain_id = f"domain:{domain_name}"
        nodes[domain_id] = {"id": domain_id, "type": "domain", "name": domain_name}

        for label in domain["labels"]:
            parts = label.split(".")
            topic_id = f"topic:{label}"
            nodes[topic_id] = {"id": topic_id, "type": "topic", "name": label}

            parent_id = domain_id
            for depth in range(2, len(parts)):
                group_name = ".".join(parts[:depth])
                group_id = f"topic_group:{group_name}"
                nodes[group_id] = {
                    "id": group_id,
                    "type": "topic_group",
                    "name": group_name,
                }
                edges.add((group_id, "child_of", parent_id))
                parent_id = group_id
            edges.add((topic_id, "child_of", parent_id))
            edges.add((topic_id, "in_domain", domain_id))

    return {
        "schema_version": "wikimedia-topics-64.1",
        "labels_as_message_edges": False,
        "nodes": sorted(nodes.values(), key=lambda node: node["id"]),
        "edges": [
            {"source": source, "relation": relation, "target": target}
            for source, relation, target in sorted(edges)
        ],
    }


def validate_targets(
    targets: list[dict[str, Any]], taxonomy: dict[str, Any]
) -> None:
    allowed = {
        label
        for domain in taxonomy["domains"]
        for label in domain["labels"]
    }
    for target in targets:
        unknown = set(target["topic_ids"]) - allowed
        if unknown:
            raise ValueError(
                f"Unknown topics for {target['dataset_article_id']}: {sorted(unknown)}"
            )


def summarize(
    evidence: dict[str, Any],
    projection: dict[str, Any],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": evidence["schema_version"],
        "article_targets": len(targets),
        "evidence_nodes": dict(
            sorted(Counter(node["type"] for node in evidence["nodes"]).items())
        ),
        "evidence_edges": dict(
            sorted(Counter(edge["relation"] for edge in evidence["edges"]).items())
        ),
        "gnn_nodes": dict(
            sorted(Counter(node["type"] for node in projection["nodes"]).items())
        ),
        "gnn_edges": dict(
            sorted(Counter(edge["relation"] for edge in projection["edges"]).items())
        ),
        "label_edge_count": sum(
            edge["relation"] == "has_topic" for edge in projection["edges"]
        ),
        "splits": dict(sorted(Counter(target["split"] for target in targets).items())),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_targets(path: Path, targets: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(target, ensure_ascii=False) + "\n")


def _short_label(node: dict[str, Any]) -> str:
    label = str(node.get("title") or node.get("name") or node["id"])
    return label if len(label) <= 24 else f"{label[:21]}…"


def write_svg(path: Path, projection: dict[str, Any]) -> None:
    """Write deterministic sampled graph visualization without extra dependencies."""
    directed = nx.DiGraph()
    for node in projection["nodes"]:
        directed.add_node(node["id"], **node)
    for edge in projection["edges"]:
        directed.add_edge(edge["source"], edge["target"], relation=edge["relation"])

    internal_articles = [
        node_id
        for node_id, data in directed.nodes(data=True)
        if data["type"] == "article" and not data.get("external_context")
    ]
    if internal_articles:
        article = min(internal_articles, key=directed.degree)
        selected = set(
            nx.single_source_shortest_path_length(
                directed.to_undirected(), article, cutoff=2
            )
        )
        directed = directed.subgraph(selected).copy()

    positions = nx.spring_layout(directed, seed=42, k=1.8)
    width, height, padding = 1600, 1000, 90

    def point(node_id: str) -> tuple[float, float]:
        x, y = positions[node_id]
        return (
            padding + (x + 1) * (width - 2 * padding) / 2,
            padding + (y + 1) * (height - 2 * padding) / 2,
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="40" y="48" font-family="sans-serif" font-size="28" '
        'font-weight="600" fill="#0f172a">KG1-A.2 pilot GNN projection</text>',
        '<text x="40" y="78" font-family="sans-serif" font-size="16" '
        'fill="#475569">Chunks are nodes; spaCy types are entity features; '
        'labels stay outside</text>',
    ]

    for source, target, data in directed.edges(data=True):
        x1, y1 = point(source)
        x2, y2 = point(target)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )
        middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2
        lines.append(
            f'<text x="{middle_x:.1f}" y="{middle_y - 5:.1f}" '
            'font-family="sans-serif" font-size="11" fill="#475569" '
            f'text-anchor="middle">{html.escape(data["relation"])}</text>'
        )

    for node_id, node in directed.nodes(data=True):
        x, y = point(node_id)
        node_type = node["type"]
        color = NODE_COLORS[node_type]
        radius = 18 if node_type == "article" else 14
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
            f'fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y + radius + 16:.1f}" '
            'font-family="sans-serif" font-size="12" fill="#0f172a" '
            f'text-anchor="middle">{html.escape(_short_label(node))}</text>'
        )

    legend_x = 40
    for index, (node_type, color) in enumerate(NODE_COLORS.items()):
        if node_type not in GNN_NODE_TYPES:
            continue
        x = legend_x + index * 170
        lines.append(
            f'<circle cx="{x}" cy="{height - 35}" r="8" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{x + 14}" y="{height - 30}" font-family="sans-serif" '
            f'font-size="13" fill="#334155">{html.escape(node_type)}</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_spacy_model(model_name: str) -> Any:
    try:
        import spacy

        nlp = spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model {model_name!r} is required. Run: "
            f"uv run python -m spacy download {model_name}"
        ) from exc
    if "ner" not in nlp.pipe_names or "parser" not in nlp.pipe_names:
        raise SystemExit(
            f"spaCy model {model_name!r} must include both parser and ner components"
        )
    return nlp


def build(input_path: Path, taxonomy_path: Path, output_dir: Path, model_name: str) -> None:
    records = read_jsonl(input_path)
    taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    nlp = load_spacy_model(model_name)

    graph = EvidenceGraph()
    targets = [extract_article(record, nlp, graph) for record in records]
    validate_targets(targets, taxonomy)

    evidence = graph.as_dict()
    validate_evidence_graph(evidence)
    projection = project_for_gnn(evidence)
    label_graph = build_label_graph(taxonomy)
    summary = summarize(evidence, projection, targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evidence_graph.json", evidence)
    write_json(output_dir / "gnn_projection.json", projection)
    write_json(output_dir / "label_graph.json", label_graph)
    write_targets(output_dir / "targets.jsonl", targets)
    write_json(output_dir / "summary.json", summary)
    write_svg(output_dir / "visualization.svg", projection)

    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build KG1-A.2 artifacts")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--model", default="en_core_web_sm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args.input, args.taxonomy, args.output, args.model)


if __name__ == "__main__":
    main()
