"""Experiment configuration — parsed from a YAML file.

Usage:
    from polygraph.benchmark_pipeline.config import ExperimentConfig

    config = ExperimentConfig.from_yaml("experiments/001_baseline/config.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from polygraph._shared.stage_config import (
    BuildConfig,
    ExtractionConfig,
    ResolutionConfig,
)


@dataclass
class ExperimentConfig:
    """All parameters needed to run a benchmark experiment.

    Load from a YAML file via ``ExperimentConfig.from_yaml(path)``.
    The YAML must contain at minimum a ``name`` and ``pipeline.variant``.
    All other fields have sensible defaults.
    """

    name: str
    description: str = ""
    pipeline_variant: str = "baseline"
    input_paths: list[Path] = field(default_factory=list)
    output_dir: Path = Path("outputs")
    ontology_path: Path | None = None
    dataset_name: str | None = None
    dataset_params: dict[str, Any] = field(default_factory=dict)
    upload_neo4j: bool = False
    clear_neo4j: bool = False
    llm_judge: bool = False
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Parse an experiment YAML file into a validated config object.

        The YAML structure expected::

            name: "My Experiment"
            description: "What this experiment tests"

            pipeline:
              variant: "baseline"
              extraction:
                mode: "composed"
                entity_method: "spacy"
                relation_method: "ontology_rules"
              resolution:
                method: "string"
              build:
                method: "networkx"

            input:
              paths:
                - "data/wikipedia/"

            output:
              dir: "outputs/"          # optional — defaults to <experiment_dir>/outputs/

            ontology: "configs/default_ontology.yaml"   # optional

            neo4j:
              upload: false
              clear: false

            evaluation:
              llm_judge: false

        """
        raw = _load_yaml(path)

        name = raw.get("name")
        if not name:
            raise ValueError(f"Experiment config missing required 'name' field: {path}")

        description = raw.get("description", "")

        pipeline_block = raw.get("pipeline", {})
        pipeline_variant = pipeline_block.get("variant", "baseline")

        input_block = raw.get("input", {})
        input_paths = [Path(p) for p in input_block.get("paths", [])]
        if not input_paths:
            raise ValueError(f"Experiment config missing 'input.paths': {path}")

        # Optional dataset — auto-download via polygraph.data.Data
        dataset_block = input_block.get("dataset", {})
        dataset_name = dataset_block.get("name") if dataset_block else None
        dataset_params = dataset_block.get("params", {}) if dataset_block else {}

        output_block = raw.get("output", {})
        config_dir = Path(path).resolve().parent
        output_dir_raw = output_block.get("dir")
        output_dir = Path(output_dir_raw) if output_dir_raw else config_dir / "results"

        ontology_path = None
        ontology_raw = raw.get("ontology")
        if ontology_raw:
            ontology_path = Path(ontology_raw)

        neo4j_block = raw.get("neo4j", {})
        upload_neo4j = neo4j_block.get("upload", False)
        clear_neo4j = neo4j_block.get("clear", False)

        eval_block = raw.get("evaluation", {})
        llm_judge = eval_block.get("llm_judge", False)

        # Parse pipeline block into typed stage configs
        pipeline_block = raw.get("pipeline", {})
        extraction = ExtractionConfig.from_dict(pipeline_block.get("extraction"))
        resolution = ResolutionConfig.from_dict(pipeline_block.get("resolution"))
        build = BuildConfig.from_dict(pipeline_block.get("build"))

        # Collect any unrecognized top-level keys as extras for pipeline kwargs.
        known_keys = {
            "name",
            "description",
            "pipeline",
            "input",
            "output",
            "ontology",
            "neo4j",
            "evaluation",
        }
        extra = {k: v for k, v in raw.items() if k not in known_keys}

        return cls(
            name=name,
            description=description,
            pipeline_variant=pipeline_variant,
            input_paths=input_paths,
            output_dir=output_dir,
            ontology_path=ontology_path,
            dataset_name=dataset_name,
            dataset_params=dataset_params,
            upload_neo4j=upload_neo4j,
            clear_neo4j=clear_neo4j,
            llm_judge=llm_judge,
            extraction=extraction,
            resolution=resolution,
            build=build,
            extra=extra,
        )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file is empty or missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}
