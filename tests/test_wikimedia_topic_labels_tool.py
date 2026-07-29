import bz2
import importlib.util
import json
from pathlib import Path

import yaml

TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "data_retrieval"
    / "prepare_wikimedia_topic_labels.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_wikimedia_topic_labels", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _write_definitions(tmp_path: Path) -> tuple[Path, Path]:
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        yaml.safe_dump(
            {
                "label_count": 2,
                "domains": [
                    {
                        "id": "Example",
                        "labels": ["Example.Topic A", "Example.Topic B"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    files = {
        key: {
            "id": index,
            "name": f"{key}.json",
            "url": f"https://example.invalid/{key}",
            "size": 1,
            "md5": "0" * 32,
        }
        for index, key in enumerate(
            ("taxonomy_mapping", "template_mapping", "labels_en"),
            start=1,
        )
    }
    manifest_path.write_text(
        yaml.safe_dump({"source": {"version": 4}, "files": files}),
        encoding="utf-8",
    )
    return manifest_path, taxonomy_path


def test_wikimedia_definitions_validate_pinned_files_and_labels(tmp_path):
    manifest_path, taxonomy_path = _write_definitions(tmp_path)

    result = TOOL.validate_definitions(manifest_path, taxonomy_path)

    assert result == {
        "taxonomy_labels": 2,
        "manifest_files": ["labels_en", "taxonomy_mapping", "template_mapping"],
        "source_version": 4,
        "status": "valid",
    }


def test_wikimedia_normalization_is_qid_grouped_and_multilabel():
    source = {
        "title": "Example",
        "qid": "Q42",
        "sitelinks": {"enwiki": 42},
        "topics": ["Example.Topic A", "Example.Topic B", "Example.Topic A"],
        "article_revid": 100,
        "talk_pid": 200,
        "talk_revid": 300,
        "wp_templates": ["Example project"],
    }

    first = TOOL.normalize_record(source, {"Example.Topic A", "Example.Topic B"})
    second = TOOL.normalize_record(source, {"Example.Topic A", "Example.Topic B"})

    assert first is not None
    assert first == second
    assert first["article_id"] == "wikipedia:en:42"
    assert first["topic_ids"] == ["Example.Topic A", "Example.Topic B"]
    assert first["split"] == TOOL.deterministic_split("Q42")


def test_wikimedia_prepare_writes_records_and_summary(tmp_path):
    _, taxonomy_path = _write_definitions(tmp_path)
    input_path = tmp_path / "labels.json.bz2"
    output_path = tmp_path / "prepared" / "labels.jsonl"
    summary_path = tmp_path / "prepared" / "summary.json"
    records = [
        {
            "title": "One",
            "qid": "Q1",
            "sitelinks": {"enwiki": 1},
            "topics": ["Example.Topic A"],
            "article_revid": 10,
            "talk_pid": 11,
            "talk_revid": 12,
            "wp_templates": ["Project A"],
        },
        {
            "title": "Two",
            "qid": "Q2",
            "sitelinks": {"enwiki": 2},
            "topics": ["Example.Topic B"],
            "article_revid": 20,
            "talk_pid": 21,
            "talk_revid": 22,
            "wp_templates": ["Project B"],
        },
    ]
    with bz2.open(input_path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    result = TOOL.prepare(
        input_path,
        output_path,
        summary_path,
        taxonomy_path,
    )

    prepared = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(prepared) == 2
    assert result["records"] == 2
    assert result["labels_observed"] == 2
    assert result["labels_missing"] == []
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result
