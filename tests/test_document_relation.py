"""Tests for document-to-document relation extraction."""

from polygraph._shared import Document, entity_id
from polygraph.kg_build.extract.document_relation._base import DocumentRelationExtractor
from polygraph.kg_build.extract.document_relation.citation import (
    CitationExtractor,
)
from polygraph.kg_build.extract.document_relation.composite import (
    CompositeDocRelationExtractor,
)
from polygraph.kg_build.extract.document_relation.hyperlink import (
    HyperlinkExtractor,
)
from polygraph.kg_build.extract.document_relation.registry import (
    create_doc_relation_method,
)
from polygraph.kg_build.extract.document_relation.series import (
    SeriesExtractor,
)
from polygraph.kg_build.extract.document_relation.shared_authors import (
    SharedAuthorsExtractor,
)
from polygraph.kg_build.extract.document_relation.shared_references import (
    SharedReferencesExtractor,
)

# ── Helpers ────────────────────────────────────────────────────


def _make_doc(doc_id: str, content: str = "", **metadata) -> Document:
    """Create a Document with the given id and metadata."""
    return Document(
        content=content,
        source=f"source/{doc_id}",
        doc_id=doc_id,
        metadata=metadata,
    )


def _doc_eid(doc_id: str) -> str:
    return entity_id("Document", doc_id)


# ── HyperlinkExtractor ─────────────────────────────────────────


def test_hyperlink_extracts_entities_for_all_docs():
    docs = [
        _make_doc("doc_a", "Hello", url="https://a.com"),
        _make_doc("doc_b", "World", url="https://b.com"),
    ]
    entities, triples = HyperlinkExtractor().extract(docs)
    ids = {e["id"] for e in entities}
    assert _doc_eid("doc_a") in ids
    assert _doc_eid("doc_b") in ids
    assert all(e["type"] == "Document" for e in entities)


def test_hyperlink_detects_url_in_content():
    docs = [
        _make_doc("doc_a", "See https://b.com/article for more", url="https://a.com"),
        _make_doc("doc_b", "Content here", url="https://b.com/article"),
    ]
    _, triples = HyperlinkExtractor().extract(docs)
    assert len(triples) == 1
    assert triples[0][0] == _doc_eid("doc_a")
    assert triples[0][1] == "hyperlinks_to"
    assert triples[0][2] == _doc_eid("doc_b")


def test_hyperlink_no_self_links():
    docs = [
        _make_doc("doc_a", "See https://a.com/self", url="https://a.com/self"),
    ]
    _, triples = HyperlinkExtractor().extract(docs)
    assert len(triples) == 0


def test_hyperlink_no_url_no_triples():
    docs = [
        _make_doc("doc_a", "No URLs here", url="https://a.com"),
        _make_doc("doc_b", "Also no URLs", url="https://b.com"),
    ]
    _, triples = HyperlinkExtractor().extract(docs)
    assert triples == []


def test_hyperlink_docs_without_url_still_get_entities():
    """Documents without a url field still produce Document entities."""
    docs = [_make_doc("plain_doc", "No url metadata")]
    entities, triples = HyperlinkExtractor().extract(docs)
    assert len(entities) == 1
    assert entities[0]["id"] == _doc_eid("plain_doc")


# ── SharedAuthorsExtractor ─────────────────────────────────────


def test_shared_authors_with_overlap():
    docs = [
        _make_doc("doc_a", authors=["Alice Smith", "Bob Jones"]),
        _make_doc("doc_b", authors=["Bob Jones", "Carol Lee"]),
    ]
    _, triples = SharedAuthorsExtractor().extract(docs)
    assert len(triples) == 1
    assert triples[0][1] == "shares_authors_with"


def test_shared_authors_no_overlap():
    docs = [
        _make_doc("doc_a", authors=["Alice Smith"]),
        _make_doc("doc_b", authors=["Bob Jones"]),
    ]
    _, triples = SharedAuthorsExtractor().extract(docs)
    assert triples == []


def test_shared_authors_comma_separated_string():
    docs = [
        _make_doc("doc_a", authors="Alice Smith, Bob Jones"),
        _make_doc("doc_b", authors="Bob Jones, Carol Lee"),
    ]
    _, triples = SharedAuthorsExtractor().extract(docs)
    assert len(triples) == 1


def test_shared_authors_case_insensitive():
    docs = [
        _make_doc("doc_a", authors=["alice smith"]),
        _make_doc("doc_b", authors=["ALICE SMITH"]),
    ]
    _, triples = SharedAuthorsExtractor().extract(docs)
    assert len(triples) == 1


def test_shared_authors_skips_docs_without_authors():
    """A doc with no authors field still gets an entity but no triple."""
    docs = [
        _make_doc("doc_a", authors=["Alice"]),
        _make_doc("doc_b"),  # no authors
    ]
    entities, triples = SharedAuthorsExtractor().extract(docs)
    assert len(entities) == 2  # both get Document nodes
    assert triples == []


# ── SeriesExtractor ────────────────────────────────────────────


def test_series_ordered_by_index():
    docs = [
        _make_doc("doc_a", series="Tutorial", series_index=1),
        _make_doc("doc_b", series="Tutorial", series_index=2),
        _make_doc("doc_c", series="Tutorial", series_index=3),
    ]
    _, triples = SeriesExtractor().extract(docs)
    assert len(triples) == 2  # 1→2, 2→3
    assert triples[0] == (_doc_eid("doc_a"), "is_part_of_series", _doc_eid("doc_b"), "tutorial", "")
    assert triples[1] == (_doc_eid("doc_b"), "is_part_of_series", _doc_eid("doc_c"), "tutorial", "")


def test_series_ordered_by_date():
    docs = [
        _make_doc("doc_a", series="Journal", upload_date="2024-01"),
        _make_doc("doc_b", series="Journal", upload_date="2024-02"),
    ]
    _, triples = SeriesExtractor().extract(docs)
    assert len(triples) == 1
    assert triples[0][0] == _doc_eid("doc_a")  # earlier → later


def test_series_no_ordering_clique():
    """When no index or date, documents in same series form a clique."""
    docs = [
        _make_doc("doc_a", series="Misc"),
        _make_doc("doc_b", series="Misc"),
        _make_doc("doc_c", series="Misc"),
    ]
    _, triples = SeriesExtractor().extract(docs)
    assert len(triples) == 3  # 3 choose 2


def test_series_single_doc_no_edge():
    docs = [_make_doc("doc_a", series="Solo")]
    _, triples = SeriesExtractor().extract(docs)
    assert triples == []


def test_series_different_series_no_edge():
    docs = [
        _make_doc("doc_a", series="A"),
        _make_doc("doc_b", series="B"),
    ]
    _, triples = SeriesExtractor().extract(docs)
    assert triples == []


# ── CompositeDocRelationExtractor ──────────────────────────────


def test_composite_combines_multiple_doc_extractors():
    docs = [
        _make_doc("doc_a", "link: https://b.com", url="https://a.com", authors=["Alice"]),
        _make_doc("doc_b", "content", url="https://b.com", authors=["Alice"]),
    ]
    composite = CompositeDocRelationExtractor(
        methods=[],
    )
    # Manually inject extractors to avoid registry dependency in unit test
    composite._extractors = [
        HyperlinkExtractor(),
        SharedAuthorsExtractor(),
    ]
    entities, triples = composite.extract(docs)

    # Both docs appear exactly once
    eids = [e["id"] for e in entities]
    assert eids.count(_doc_eid("doc_a")) == 1
    assert eids.count(_doc_eid("doc_b")) == 1

    # Both hyperlink and shared_authors triples present
    predicates = {t[1] for t in triples}
    assert "hyperlinks_to" in predicates
    assert "shares_authors_with" in predicates


def test_composite_deduplicates_overlapping_triples():
    """Two extractors producing the same triple → only one copy."""
    docs = [
        _make_doc("doc_a", authors=["Alice"]),
        _make_doc("doc_b", authors=["Alice"]),
    ]

    class _FakeExtractor(DocumentRelationExtractor):
        def extract(self, documents):
            entities = [
                {"id": _doc_eid(d.doc_id), "name": d.doc_id, "type": "Document"} for d in documents
            ]
            triples = [(_doc_eid("doc_a"), "shares_authors_with", _doc_eid("doc_b"), "Alice", "")]
            return entities, triples

    composite = CompositeDocRelationExtractor(methods=[])
    composite._extractors = [_FakeExtractor(), _FakeExtractor()]
    _, triples = composite.extract(docs)
    assert len(triples) == 1


def test_composite_empty_methods_list():
    composite = CompositeDocRelationExtractor(methods=[])
    entities, triples = composite.extract([])
    assert entities == []
    assert triples == []


# ── Registry ───────────────────────────────────────────────────


def test_doc_registry_instantiates_all_methods():
    """Each registered method can be instantiated via the registry."""
    for name in [
        "hyperlink",
        "shared_authors",
        "series",
        "composite",
        "citation",
        "shared_references",
    ]:
        if name == "composite":
            extractor = create_doc_relation_method(name, methods=["hyperlink"])
        elif name == "shared_references":
            extractor = create_doc_relation_method(name, threshold=0.0)
        else:
            extractor = create_doc_relation_method(name)
        assert isinstance(extractor, DocumentRelationExtractor), f"{name} failed"


def test_doc_registry_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown document relation method"):
        create_doc_relation_method("nonexistent")


# ── CitationExtractor ──────────────────────────────────────────


def test_citation_by_doi():
    docs = [
        _make_doc("paper_a", "We build on prior work [1].", doi="10.1234/abc"),
        _make_doc("paper_b", "This paper references 10.1234/abc heavily.", doi="10.5678/xyz"),
    ]
    _, triples = CitationExtractor().extract(docs)
    assert len(triples) == 1
    assert triples[0][0] == _doc_eid("paper_b")
    assert triples[0][1] == "cites"
    assert triples[0][2] == _doc_eid("paper_a")


def test_citation_by_title():
    docs = [
        _make_doc("paper_a", "Content", title="Attention Is All You Need"),
        _make_doc("paper_b", "As shown in Attention Is All You Need, transformers..."),
    ]
    _, triples = CitationExtractor().extract(docs)
    assert len(triples) >= 1
    predicates = {t[1] for t in triples}
    assert "cites" in predicates


def test_citation_by_author_year():
    docs = [
        _make_doc("paper_a", "Content", authors=["Vaswani"], year="2017"),
        _make_doc(
            "paper_b", "Transformers (Vaswani, 2017) revolutionized NLP.", authors=["Devlin"]
        ),
    ]
    _, triples = CitationExtractor().extract(docs)
    cites_triples = [t for t in triples if t[1] == "cites"]
    assert any(t[2] == _doc_eid("paper_a") for t in cites_triples)


def test_citation_deduplication():
    """Multiple citation patterns matching the same target → one edge."""
    docs = [
        _make_doc("paper_a", "Content", doi="10.1234/abc", title="The Great Paper"),
        _make_doc("paper_b", "10.1234/abc and The Great Paper both cited.", doi="10.5678/xyz"),
    ]
    _, triples = CitationExtractor().extract(docs)
    cites_triples = [t for t in triples if t[1] == "cites"]
    assert len(cites_triples) == 1


def test_citation_no_self_cite():
    docs = [
        _make_doc("paper_a", "Our earlier work 10.1234/abc is extended here.", doi="10.1234/abc"),
    ]
    _, triples = CitationExtractor().extract(docs)
    assert triples == []


def test_citation_no_metadata_no_triples():
    docs = [
        _make_doc("plain_a", "No DOIs or author-year here."),
        _make_doc("plain_b", "Also plain content."),
    ]
    _, triples = CitationExtractor().extract(docs)
    assert triples == []


# ── SharedReferencesExtractor ──────────────────────────────────


def test_shared_references_with_overlap():
    docs = [
        _make_doc("doc_a", "See 10.1234/abc and 10.5678/xyz for background."),
        _make_doc("doc_b", "Building on 10.1234/abc and arxiv:2001.00001."),
    ]
    _, triples = SharedReferencesExtractor().extract(docs)
    assert len(triples) == 1
    assert triples[0][1] == "shares_reference_with"
    assert "10.1234/abc" in triples[0][3]


def test_shared_references_no_overlap():
    docs = [
        _make_doc("doc_a", "See 10.1234/abc."),
        _make_doc("doc_b", "See 10.9999/zzz."),
    ]
    _, triples = SharedReferencesExtractor().extract(docs)
    assert triples == []


def test_shared_references_with_arxiv():
    docs = [
        _make_doc("doc_a", "See arxiv:2001.00001 for details."),
        _make_doc("doc_b", "Also see arxiv:2001.00001 and arxiv:2101.00002."),
    ]
    _, triples = SharedReferencesExtractor().extract(docs)
    assert len(triples) == 1
    assert "arxiv:2001.00001" in triples[0][3]


def test_shared_references_threshold_filters():
    """With threshold=1.0 (exact match), partial overlap is ignored."""
    docs = [
        _make_doc("doc_a", "See 10.1234/abc and 10.5678/xyz."),
        _make_doc("doc_b", "See 10.1234/abc and 10.9999/zzz."),
    ]
    _, triples = SharedReferencesExtractor(threshold=1.0).extract(docs)
    assert triples == []

    # threshold=0.0 (any overlap) → edge
    _, triples2 = SharedReferencesExtractor(threshold=0.0).extract(docs)
    assert len(triples2) == 1


def test_shared_references_single_doc_no_edge():
    docs = [_make_doc("doc_a", "See 10.1234/abc and 10.5678/xyz.")]
    _, triples = SharedReferencesExtractor().extract(docs)
    assert triples == []


def test_shared_references_no_refs_no_edges():
    docs = [
        _make_doc("doc_a", "No references here."),
        _make_doc("doc_b", "Nothing to cite."),
    ]
    _, triples = SharedReferencesExtractor().extract(docs)
    assert triples == []
