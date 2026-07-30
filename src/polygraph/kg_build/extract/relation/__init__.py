"""Relation extraction methods."""

from .composite import CompositeRelationExtractor
from .ontology_rules import OntologyRuleRelationExtractor
from .structured_llm import StructuredLLMRelationExtractor

__all__ = [
    "CompositeRelationExtractor",
    "OntologyRuleRelationExtractor",
    "StructuredLLMRelationExtractor",
]
