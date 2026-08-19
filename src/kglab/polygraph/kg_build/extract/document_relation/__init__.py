"""Document-to-document relation extraction methods.

Each extractor implements :class:`DocumentRelationExtractor` and operates
on the full batch of source documents (not individual chunks). This enables
cross-document relations like hyperlinks, citations, shared authors, etc.

Usage::

    from kglab.kg_build.extract.document_relation import (
        HyperlinkExtractor,
        SharedAuthorsExtractor,
        DocumentRelationExtractor,
    )

    ext = HyperlinkExtractor(ontology)
    doc_entities, triples = ext.extract(documents)
"""

from ._base import DocumentRelationExtractor
from .citation import CitationExtractor
from .composite import CompositeDocRelationExtractor
from .hyperlink import HyperlinkExtractor
from .registry import DOC_RELATION_METHODS, create_doc_relation_method
from .series import SeriesExtractor
from .shared_authors import SharedAuthorsExtractor
from .shared_references import SharedReferencesExtractor

__all__ = [
    "CitationExtractor",
    "CompositeDocRelationExtractor",
    "DOC_RELATION_METHODS",
    "DocumentRelationExtractor",
    "HyperlinkExtractor",
    "SeriesExtractor",
    "SharedAuthorsExtractor",
    "SharedReferencesExtractor",
    "create_doc_relation_method",
]
