"""Lazy registry for document relation extraction methods."""

from importlib import import_module
from typing import Any

DOC_RELATION_METHODS: dict[str, str] = {
    "citation": ("polygraph.kg_build.extract.document_relation.citation:CitationExtractor"),
    "composite": (
        "polygraph.kg_build.extract.document_relation.composite:CompositeDocRelationExtractor"
    ),
    "hyperlink": ("polygraph.kg_build.extract.document_relation.hyperlink:HyperlinkExtractor"),
    "series": ("polygraph.kg_build.extract.document_relation.series:SeriesExtractor"),
    "shared_authors": (
        "polygraph.kg_build.extract.document_relation.shared_authors:SharedAuthorsExtractor"
    ),
    "shared_references": (
        "polygraph.kg_build.extract.document_relation.shared_references:SharedReferencesExtractor"
    ),
}


def _create(registry: dict[str, str], method: str, **kwargs: Any) -> Any:
    try:
        import_path = registry[method]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown document relation method '{method}'. Available: {choices}"
        ) from exc
    module_name, class_name = import_path.split(":", maxsplit=1)
    method_type = getattr(import_module(module_name), class_name)
    return method_type(**kwargs)


def create_doc_relation_method(method: str, **kwargs: Any) -> Any:
    """Instantiate a document relation extractor by name."""
    return _create(DOC_RELATION_METHODS, method, **kwargs)
