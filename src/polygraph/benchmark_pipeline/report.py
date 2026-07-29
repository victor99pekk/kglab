"""Benchmark result types and report writing.

Writes a ``results_summary.json`` that captures the full experiment context —
config snapshot, metrics, structural audit, artifact paths, and Neo4j status —
so every experiment is self-documenting and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Aggregated results from a single benchmark run."""

    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_variant: str = ""
    input_paths: list[str] = field(default_factory=list)
    output_dir: str = ""

    # Metrics loaded from the pipeline's metrics.json output
    metrics: dict[str, Any] = field(default_factory=dict)
    structural_audit: dict[str, Any] = field(default_factory=dict)

    # Paths to generated artifacts (relative to output_dir for portability)
    artifacts: dict[str, str] = field(default_factory=dict)

    # Neo4j upload status
    neo4j_uploaded: bool = False
    neo4j_cleared: bool = False

    # Any extra metadata
    config_snapshot: dict[str, Any] = field(default_factory=dict)


def write_report(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Serialize a ``BenchmarkResult`` as ``results_summary.json``.

    Args:
        result: The completed benchmark result to write.
        output_dir: Directory to write the report into (created if needed).

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "results_summary.json"

    payload = asdict(result)
    # Make relative artifact paths POSIX-style for cross-platform portability
    if payload.get("artifacts"):
        payload["artifacts"] = {
            k: str(v).replace("\\", "/") for k, v in payload["artifacts"].items()
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)

    return report_path


def _coerce_relative(path: str | Path, base_dir: str | Path) -> str:
    """Return *path* relative to *base_dir* if it's inside it, else absolute."""
    try:
        rel = Path(path).resolve().relative_to(Path(base_dir).resolve())
        return str(rel)
    except ValueError:
        return str(path)
