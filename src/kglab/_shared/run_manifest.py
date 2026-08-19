"""Run manifest — captures the full pipeline architecture and hyperparameters for every run.

Serialized to ``run_manifest.yaml`` in the output directory so every generated KG
is fully reproducible and auditable.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _find_next_run_index(base_dir: Path) -> int:
    """Scan ``base_dir`` for ``KG_0``, ``KG_1``, … and return the next available index."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing: list[int] = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.startswith("KG_"):
            with contextlib.suppress(ValueError):
                existing.append(int(child.name.split("_", 1)[1]))
    return max(existing, default=-1) + 1


def next_run_dir(base_dir: str | Path = "generated_KGs") -> Path:
    """Return the next auto-incremented run directory (e.g. ``generated_KGs/KG_3``).

    Args:
        base_dir: Root directory for generated KGs.  Created if it doesn't exist.

    Returns:
        Absolute path to the new run directory.  The directory is NOT created —
        the pipeline's ``output_dir.mkdir(parents=True)`` handles that.
    """
    root = Path(base_dir)
    idx = _find_next_run_index(root)
    return root / f"KG_{idx}"


@dataclass
class RunManifest:
    """Full record of one pipeline execution — all hyperparameters, inputs, and results.

    Serialized to YAML in the output directory.  Custom pipelines can add
    arbitrary keys via the ``extra`` dict.

    Usage::

        manifest = RunManifest(
            pipeline_class="Baseline",
            input_paths=["data/wikipedia/connected.jsonl"],
            output_dir="generated_KGs/KG_0",
            preprocess=asdict(preprocess_config),
            extraction=asdict(extraction_config),
        )
        manifest.dump_yaml(output_dir / "run_manifest.yaml")
    """

    pipeline_class: str
    input_paths: list[str]
    output_dir: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # ── Stage configs (all optional — not every pipeline uses every stage) ──
    preprocess: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    resolution: dict[str, Any] | None = None
    build: dict[str, Any] | None = None
    linking: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    export: dict[str, Any] | None = None
    ontology: dict[str, Any] | None = None

    # ── Run metadata ──────────────────────────────────────────
    run_index: int | None = None
    """Auto-incremented index under ``generated_KGs/`` (e.g. 0 for ``KG_0``)."""

    run_duration_seconds: float | None = None
    """Wall-clock duration of the full pipeline execution."""

    # ── Results summary ───────────────────────────────────────
    results: dict[str, Any] = field(default_factory=dict)
    """High-level counts and scores populated after execution
    (documents, chunks, entities, nodes, edges, overall_score, …)."""

    # ── Extensibility ─────────────────────────────────────────
    extra: dict[str, Any] = field(default_factory=dict)
    """Arbitrary extra fields for custom pipelines.  Merged at the top level
    when serialized."""

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a flat-ish dict ready for YAML serialization.

        Drops ``None`` values for cleanliness and merges ``extra`` at the top level.
        """
        d: dict[str, Any] = {
            "pipeline": {
                "class": self.pipeline_class,
            },
            "run": {
                "timestamp": self.timestamp,
                "input_paths": self.input_paths,
                "output_dir": self.output_dir,
            },
            "stages": {},
            "results": self.results,
        }

        if self.run_index is not None:
            d["run"]["run_index"] = self.run_index
        if self.run_duration_seconds is not None:
            d["run"]["run_duration_seconds"] = round(self.run_duration_seconds, 2)

        # ── Populate stages that have values ──
        stage_map: dict[str, dict[str, Any] | None] = {
            "preprocess": self.preprocess,
            "extraction": self.extraction,
            "resolution": self.resolution,
            "build": self.build,
            "linking": self.linking,
            "evaluation": self.evaluation,
            "export": self.export,
        }
        for name, value in stage_map.items():
            if value is not None:
                d["stages"][name] = value

        if self.ontology is not None:
            d["ontology"] = self.ontology

        # Merge extras at top level
        if self.extra:
            d["extra"] = self.extra

        return d

    def dump_yaml(self, path: Path) -> None:
        """Write the manifest as YAML to *path*."""
        import yaml

        d = self.to_dict()
        path.write_text(
            yaml.dump(d, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"[manifest] → {path}")

    def dump_json(self, path: Path) -> None:
        """Write the manifest as JSON to *path* (alternative to YAML)."""
        d = self.to_dict()
        path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
        print(f"[manifest] → {path}")
