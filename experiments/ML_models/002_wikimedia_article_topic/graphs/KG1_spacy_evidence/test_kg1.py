"""Experiment-local checks for KG1-A."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import spacy

MODULE_PATH = Path(__file__).with_name("kg1_pipeline.py")
SPEC = importlib.util.spec_from_file_location("kg1_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
kg1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kg1)

RETRIEVER_PATH = Path(__file__).with_name("retrieve_pilot.py")
RETRIEVER_SPEC = importlib.util.spec_from_file_location("retrieve_pilot", RETRIEVER_PATH)
assert RETRIEVER_SPEC and RETRIEVER_SPEC.loader
retriever = importlib.util.module_from_spec(RETRIEVER_SPEC)
RETRIEVER_SPEC.loader.exec_module(retriever)

NEO4J_EXPORT_PATH = Path(__file__).with_name("export_neo4j.py")
NEO4J_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_neo4j", NEO4J_EXPORT_PATH
)
assert NEO4J_EXPORT_SPEC and NEO4J_EXPORT_SPEC.loader
neo4j_export = importlib.util.module_from_spec(NEO4J_EXPORT_SPEC)
NEO4J_EXPORT_SPEC.loader.exec_module(neo4j_export)


def _nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(
        [
            {"label": "PERSON", "pattern": "Ada Lovelace"},
            {"label": "ORG", "pattern": "Analytical Engine Society"},
        ]
    )
    return nlp


def _record():
    return {
        "article_id": "fixture:ada",
        "page_id": 123,
        "title": "Ada Lovelace",
        "qid": "Q7259",
        "article_revision_id": 456,
        "split": "train",
        "topic_ids": ["Culture.Biography.Women", "STEM.Computing"],
        "text": (
            "Ada Lovelace wrote about the Analytical Engine. "
            "The Analytical Engine Society preserves her legacy."
        ),
        "links": [
            {
                "surface": "Analytical Engine",
                "title": "Analytical Engine",
                "qid": "Q160928",
            }
        ],
    }


def test_topics_stay_outside_evidence_and_message_edges():
    graph = kg1.EvidenceGraph()
    target = kg1.extract_article(_record(), _nlp(), graph)
    evidence = graph.as_dict()
    projection = kg1.project_for_gnn(evidence)

    assert target["topic_ids"] == ["Culture.Biography.Women", "STEM.Computing"]
    assert all("topic_ids" not in node for node in evidence["nodes"])
    assert all(edge["relation"] != "has_topic" for edge in projection["edges"])


def test_hyperlink_qid_is_canonical_entity_and_spacy_recovers_unlinked_entity():
    graph = kg1.EvidenceGraph()
    kg1.extract_article(_record(), _nlp(), graph)
    evidence = graph.as_dict()

    entities = {
        node["id"]: node
        for node in evidence["nodes"]
        if node["type"] == "entity"
    }
    assert "entity:wikidata:Q160928" in entities
    assert any(
        node["name"] == "Analytical Engine Society"
        and node["canonical_source"] == "spacy_ner"
        for node in entities.values()
    )
    assert entities["entity:wikidata:Q7259"]["canonical_source"] == "wikipedia_page"


def test_projection_removes_sentence_and_mention_nodes_but_keeps_types():
    graph = kg1.EvidenceGraph()
    kg1.extract_article(_record(), _nlp(), graph)
    projection = kg1.project_for_gnn(graph.as_dict())

    assert {node["type"] for node in projection["nodes"]} <= {
        "article",
        "entity",
        "entity_type",
    }
    relations = {edge["relation"] for edge in projection["edges"]}
    assert "has_spacy_type" in relations
    assert "mentions" in relations
    assert "describes" in relations


def test_label_graph_has_64_topics_and_no_article_nodes():
    taxonomy = kg1.yaml.safe_load(
        kg1.DEFAULT_TAXONOMY.read_text(encoding="utf-8")
    )
    label_graph = kg1.build_label_graph(taxonomy)

    topics = [node for node in label_graph["nodes"] if node["type"] == "topic"]
    assert len(topics) == 64
    assert all(node["type"] != "article" for node in label_graph["nodes"])


def test_revision_html_parser_keeps_prose_anchor_surfaces_and_drops_infobox():
    parsed_html = """
    <table class="infobox"><tr><td><a href="/wiki/Noise" title="Noise">Noise</a></td></tr></table>
    <p>Ada studied the <a href="/wiki/Analytical_Engine" title="Analytical Engine">
    Analytical Engine</a>.</p>
    <p>She worked in <a href="/wiki/London" title="London">London</a>.<sup>[1]</sup></p>
    """

    text, links = retriever.extract_text_and_links(parsed_html)

    assert "Noise" not in text
    assert "Analytical Engine" in text
    assert "[1]" not in text
    assert links == [
        {"surface": "Analytical Engine", "title": "Analytical Engine"},
        {"surface": "London", "title": "London"},
    ]


def test_neo4j_adapter_matches_repository_contract_without_label_edges():
    graph = kg1.EvidenceGraph()
    kg1.extract_article(_record(), _nlp(), graph)
    projection = kg1.project_for_gnn(graph.as_dict())

    adapted = neo4j_export.adapt_graph(projection)

    assert set(adapted) == {"metadata", "graph", "entities", "triples", "stats"}
    assert len(adapted["graph"]["nodes"]) == len(projection["nodes"])
    assert len(adapted["graph"]["edges"]) == len(projection["edges"])
    assert all("predicates" in edge for edge in adapted["graph"]["edges"])
    assert all(
        "HAS_TOPIC" not in edge["predicates"]
        for edge in adapted["graph"]["edges"]
    )
    assert adapted["metadata"]["labels_as_message_edges"] is False
