"""Relation extraction methods."""

from .ontology_rules import OntologyRuleRelationExtractor
from .structured_llm import StructuredLLMRelationExtractor

__all__ = ["OntologyRuleRelationExtractor", "StructuredLLMRelationExtractor"]
