"""Relation extraction methods."""

from .composite import CompositeRelationExtractor
from .en.ontology_rules import OntologyRuleRelationExtractor
from .en.structured_llm import StructuredLLMRelationExtractor

__all__ = [
    "CompositeRelationExtractor",
    "OntologyRuleRelationExtractor",
    "StructuredLLMRelationExtractor",
]
