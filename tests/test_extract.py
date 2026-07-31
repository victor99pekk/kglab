"""Tests for entity and relation extraction."""

from pathlib import Path
from types import SimpleNamespace

from polygraph._shared import Ontology
from polygraph.benchmark_pipeline.config import ExperimentConfig
from polygraph.kg_build.extract._base import Entity
from polygraph.kg_build.extract.entity import SimpleExtractor
from polygraph.kg_build.extract.joint import GraphGenExtractor
from polygraph.kg_build.extract.registry import (
    create_entity_method,
    create_joint_method,
    create_relation_method,
)
from polygraph.kg_build.extract.relation.en.ontology_rules import (
    OntologyRuleRelationExtractor,
)

_ONTOLOGY_PATH = Path(__file__).parents[1] / "configs" / "default_ontology.yaml"


def _load_test_ontology() -> Ontology:
    return Ontology.from_yaml(_ONTOLOGY_PATH)


def test_simple_extractor_captures_capitalized():
    extractor = SimpleExtractor()
    text = "Alice and Bob visited New York City last summer."
    entities = extractor.extract(text)

    names = {e.name for e in entities}
    # SimpleExtractor with CAPITALIZED_PHRASE will find multi-word capitalized phrases
    assert (
        "New York City" in names
        or any("Alice" in n for n in names)
        or any("Bob" in n for n in names)
    )


def test_extraction_registry_loads_each_method_family():
    ontology = _load_test_ontology()

    assert isinstance(create_entity_method("regex"), SimpleExtractor)
    assert isinstance(
        create_relation_method("ontology_rules", ontology=ontology),
        OntologyRuleRelationExtractor,
    )
    assert isinstance(
        create_joint_method("graphgen", ontology=ontology),
        GraphGenExtractor,
    )


def test_experiment_config_passes_stage_method_selection_to_pipeline():
    config = ExperimentConfig.from_yaml(
        Path(__file__).parents[1] / "experiments" / "kg" / "001_baseline" / "config.yaml"
    )

    assert config.extraction.entity_method == "spacy"
    assert config.extraction.relation_method == "ontology_rules"
    assert config.resolution.method == "string"
    assert config.build.method == "networkx"


def test_relation_preserves_stable_ids_evidence_and_source_chunk():
    ontology = _load_test_ontology()
    alice = Entity(name="Alice", label="PERSON")
    acme = Entity(name="Acme Corp", label="ORG")

    relations = OntologyRuleRelationExtractor(ontology=ontology).extract(
        "Alice works at Acme Corp. Another sentence.",
        [alice, acme],
        source_chunk_id="chunk:123",
    )

    assert len(relations) == 1
    subject, predicate, object_, evidence, source_chunk_id = relations[0]
    assert subject == alice.id
    assert object_ == acme.id
    assert predicate == "works_at"
    assert evidence == "Alice works at Acme Corp."
    assert source_chunk_id == "chunk:123"


def test_ontology_rule_extractor_does_not_fallback_for_unmatched_types():
    ontology = _load_test_ontology()
    alice = Entity(name="Alice", label="UNMAPPED_A")
    bob = Entity(name="Bob", label="UNMAPPED_B")

    relations = OntologyRuleRelationExtractor(ontology=ontology).extract(
        "Alice met Bob.",
        [alice, bob],
        source_chunk_id="chunk:123",
    )

    assert relations == []


def test_structural_relations_are_not_extraction_patterns():
    ontology = _load_test_ontology()

    predicates = {predicate for _, _, predicate, _ in ontology.get_relation_patterns()}

    assert predicates.isdisjoint({"appears_in", "part_of", "next"})


class _FakeDeepSeekClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_graphgen_jointly_extracts_entities_and_descriptive_relationships():
    ontology = _load_test_ontology()
    text = "Alice founded Acme Corp in Hanoi."
    extraction = """("entity"<|>"Alice"<|>"person"<|>"A founder.")##
("entity"<|>"Acme Corp"<|>"org"<|>"A company.")##
("entity"<|>"Hanoi"<|>"gpe"<|>"A city.")##
("relationship"<|>"Alice"<|>"Acme Corp"<|>"Alice founded Acme Corp.")##
("content_keywords"<|>"company founding")<|COMPLETE|>"""
    client = _FakeDeepSeekClient([extraction, "NO"])

    entities, relationships = GraphGenExtractor(ontology=ontology, client=client).extract(
        text, source_chunk_id="chunk:123"
    )

    assert [(entity.name, entity.label) for entity in entities] == [
        ("Alice", "PERSON"),
        ("Acme Corp", "ORG"),
        ("Hanoi", "GPE"),
    ]
    assert len(relationships) == 1
    relation = relationships[0]
    assert relation[1] == "RELATION"
    assert relation[3] == ""
    assert relation[4] == "chunk:123"
    assert relation[5] == "Alice founded Acme Corp."
    assert len(relation) == 6
    assert "relationship_summary" in client.calls[0]["messages"][0]["content"]
    assert "response_format" not in client.calls[0]


def test_graphgen_iteratively_gleans_missed_records():
    ontology = _load_test_ontology()
    initial = """("entity"<|>"Alice"<|>"person"<|>"A person.")<|COMPLETE|>"""
    glean = """("entity"<|>"Acme Corp"<|>"org"<|>"A company.")##
("relationship"<|>"Alice"<|>"Acme Corp"<|>"Alice founded Acme Corp.")<|COMPLETE|>"""
    client = _FakeDeepSeekClient([initial, "YES", glean, "NO"])

    entities, relationships = GraphGenExtractor(ontology=ontology, client=client).extract(
        "Source text."
    )

    assert {entity.name for entity in entities} == {"Alice", "Acme Corp"}
    assert len(relationships) == 1
    assert len(client.calls) == 4


def test_graphgen_retries_empty_response():
    ontology = _load_test_ontology()
    extraction = """("entity"<|>"Alice"<|>"person"<|>"A person.")<|COMPLETE|>"""
    client = _FakeDeepSeekClient(["", extraction, "NO"])

    entities, relationships = GraphGenExtractor(ontology=ontology, client=client).extract(
        "Some source text."
    )

    assert [entity.name for entity in entities] == ["Alice"]
    assert relationships == []
    assert len(client.calls) == 3


def test_graphgen_aggregates_repeated_descriptions_with_figure_9():
    ontology = _load_test_ontology()
    client = _FakeDeepSeekClient(
        [
            "Alice is a scientist and Nobel Prize winner.",
            "Alice founded Acme Corp and later led it.",
        ]
    )
    extractor = GraphGenExtractor(ontology=ontology, client=client, max_gleanings=0)
    resolved = [{"id": "entity:alice", "name": "Alice", "description": "first"}]
    originals = [
        {"id": "entity:alice", "description": "Alice is a scientist."},
        {"id": "entity:alice", "description": "Alice won a Nobel Prize."},
    ]
    triples = [
        ("entity:alice", "RELATION", "entity:acme", "", "chunk:1", "Alice founded Acme."),
        ("entity:alice", "RELATION", "entity:acme", "", "chunk:2", "Alice led Acme."),
    ]

    entities, relations = extractor.aggregate_descriptions(resolved, originals, {}, triples)

    assert entities[0]["description"] == "Alice is a scientist and Nobel Prize winner."
    assert {relation[5] for relation in relations} == {"Alice founded Acme Corp and later led it."}
    assert "Description List" in client.calls[0]["messages"][0]["content"]


# ── Composite relation extractor tests ─────────────────────────


def test_composite_combines_multiple_extractors():
    """Triples from all sub-extractors appear in the merged output."""
    ontology = _load_test_ontology()

    class _FakeExtractorA(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return [("a", "knows", "b", "evidence a", source_chunk_id)]

    class _FakeExtractorB(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return [("c", "works_at", "d", "evidence c", source_chunk_id)]

    composite = _make_composite_with_fakes(ontology, _FakeExtractorA, _FakeExtractorB)
    alice = Entity(name="Alice", label="PERSON")
    triples = composite.extract("text", [alice])

    keys = {(t[0], t[1], t[2]) for t in triples}
    assert ("a", "knows", "b") in keys
    assert ("c", "works_at", "d") in keys
    assert len(triples) == 2


def test_composite_deduplicates_overlapping_triples():
    """When two extractors produce the same triple, only one copy is kept."""
    ontology = _load_test_ontology()

    class _FakeExtractorA(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return [("a", "knows", "b", "evidence 1", source_chunk_id)]

    class _FakeExtractorB(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return [("a", "knows", "b", "evidence 2", source_chunk_id)]

    composite = _make_composite_with_fakes(ontology, _FakeExtractorA, _FakeExtractorB)
    triples = composite.extract("text", [Entity(name="Alice", label="PERSON")])

    assert len(triples) == 1
    # First extractor's evidence is kept (first writer wins)
    assert triples[0][3] == "evidence 1"


def test_composite_preserves_evidence_and_source_chunk():
    """Each triple keeps its 5-element format with evidence and source_chunk_id."""
    ontology = _load_test_ontology()

    class _FakeExtractorA(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return [("a", "knows", "b", "Alice knows Bob.", source_chunk_id)]

    composite = _make_composite_with_fakes(ontology, _FakeExtractorA)
    triples = composite.extract(
        "text", [Entity(name="Alice", label="PERSON")], source_chunk_id="chunk:42"
    )

    assert len(triples) == 1
    assert triples[0] == ("a", "knows", "b", "Alice knows Bob.", "chunk:42")


def test_composite_with_empty_sub_extractor():
    """An extractor returning no triples does not affect other extractors."""
    ontology = _load_test_ontology()

    class _FakeEmptyExtractor(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return []

    class _FakeExtractorA(OntologyRuleRelationExtractor):
        def extract(self, text, entities, source_chunk_id=""):
            return [("a", "knows", "b", "evidence", source_chunk_id)]

    composite = _make_composite_with_fakes(ontology, _FakeEmptyExtractor, _FakeExtractorA)
    triples = composite.extract("text", [Entity(name="Alice", label="PERSON")])

    assert len(triples) == 1
    assert triples[0][1] == "knows"


def test_composite_registry_instantiation():
    """The registry can instantiate a CompositeRelationExtractor."""
    ontology = _load_test_ontology()
    extractor = create_relation_method(
        "composite",
        ontology=ontology,
        methods=["ontology_rules"],
    )

    from polygraph.kg_build.extract.relation.composite import CompositeRelationExtractor

    assert isinstance(extractor, CompositeRelationExtractor)
    assert extractor.ontology is ontology
    assert extractor.methods == ["ontology_rules"]


# ── Helpers ────────────────────────────────────────────────────


def _make_composite_with_fakes(ontology, *fake_classes):
    """Build a CompositeRelationExtractor that uses manually instantiated fake extractors."""
    from polygraph.kg_build.extract.relation.composite import CompositeRelationExtractor

    composite = CompositeRelationExtractor(ontology, methods=[])
    composite._extractors = [cls(ontology=ontology) for cls in fake_classes]
    return composite
