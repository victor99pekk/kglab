"""Lazy registry for extraction methods used by experiments."""

from importlib import import_module
from typing import Any

ENTITY_METHODS = {
    "regex": "polygraph.kg_build.extract.entity.regex:SimpleExtractor",
    "spacy": "polygraph.kg_build.extract.entity.en.spacy:SpacyExtractor",
}

RELATION_METHODS = {
    "composite": ("polygraph.kg_build.extract.relation.composite:CompositeRelationExtractor"),
    "ontology_rules": (
        "polygraph.kg_build.extract.relation.en.ontology_rules:OntologyRuleRelationExtractor"
    ),
    "structured_llm": (
        "polygraph.kg_build.extract.relation.en.structured_llm:StructuredLLMRelationExtractor"
    ),
}

JOINT_METHODS = {
    "graphgen": "polygraph.kg_build.extract.joint.en.graphgen:GraphGenExtractor",
}


def _create(registry: dict[str, str], method: str, **kwargs: Any):
    try:
        import_path = registry[method]
    except KeyError as exc:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"Unknown extraction method '{method}'. Available: {choices}") from exc
    module_name, class_name = import_path.split(":", maxsplit=1)
    method_type = getattr(import_module(module_name), class_name)
    return method_type(**kwargs)


def create_entity_method(method: str, **kwargs: Any):
    return _create(ENTITY_METHODS, method, **kwargs)


def create_relation_method(method: str, **kwargs: Any):
    return _create(RELATION_METHODS, method, **kwargs)


def create_joint_method(method: str, **kwargs: Any):
    return _create(JOINT_METHODS, method, **kwargs)


__all__ = [
    "ENTITY_METHODS",
    "JOINT_METHODS",
    "RELATION_METHODS",
    "create_entity_method",
    "create_joint_method",
    "create_relation_method",
]
