"""Tests for typed pipeline-stage configuration."""

import pytest

from polygraph._shared.stage_config import DocumentRelationConfig, ExtractionConfig


def test_document_relations_can_be_disabled_without_specifying_methods():
    config = DocumentRelationConfig.from_dict({"enabled": False})

    assert config.enabled is False
    assert config.methods == ["hyperlink"]


def test_document_relation_config_preserves_explicit_methods():
    config = DocumentRelationConfig.from_dict(
        {
            "enabled": True,
            "methods": ["citation", "shared_references"],
            "method_options": {"shared_references": {"threshold": 0.5}},
        }
    )

    assert config.methods == ["citation", "shared_references"]
    assert config.method_options == {"shared_references": {"threshold": 0.5}}


def test_disabled_document_relations_allow_an_explicit_empty_method_list():
    config = ExtractionConfig.from_dict(
        {"document_relation": {"enabled": False, "methods": []}}
    )

    assert config.document_relation.enabled is False
    assert config.document_relation.methods == []


def test_enabled_document_relations_reject_an_empty_method_list():
    with pytest.raises(ValueError, match="at least one method when enabled"):
        DocumentRelationConfig.from_dict({"enabled": True, "methods": []})


@pytest.mark.parametrize("methods", ["hyperlink", [""], [1]])
def test_document_relation_methods_must_be_non_empty_strings(methods):
    with pytest.raises(ValueError, match="list of non-empty strings"):
        DocumentRelationConfig.from_dict({"enabled": False, "methods": methods})
