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

from kglab.benchmark_pipeline.render import STAGE_META, format_stage


@dataclass
class BenchmarkResult:
    """Aggregated results from a single benchmark run.

    Access the overall quality score directly::

        result = runner.run()
        print(f"Score: {result.overall_score:.2f}")
        print(f"Entities: {result.num_entities}")
        print(f"Triples: {result.num_triples}")
    """

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

    @property
    def overall_score(self) -> float:
        """Convenience accessor for the overall quality score (0–1)."""
        return float(self.metrics.get("overall_score", 0.0))

    @property
    def num_entities(self) -> int:
        """Number of entities in the KG."""
        return int(self.metrics.get("num_entities", 0))

    @property
    def num_triples(self) -> int:
        """Number of triples (edges) in the KG."""
        return int(self.metrics.get("num_triples", 0))


@dataclass
class StageResult:
    """Results of benchmarking one pipeline stage against a gold dataset.

    Returned by the stage runners (``Benchmark.Dedup(...).run(...)`` etc.).
    Wraps the per-pipeline metrics with a readable ``str()`` and a couple of
    convenience accessors — nothing else::

        result = Benchmark.Dedup(dataset=...).run(pipelines={...})
        print(result)              # aligned per-pipeline table
        result.best_pipeline()     # pipeline with the best headline metric
        result.to_dict()           # plain {pipeline: metrics} dict (JSON-safe)
    """

    stage: str
    """Stage name (e.g. ``"dedup"``) — must be a ``render.STAGE_META`` key."""

    results: dict[str, dict[str, Any]]
    """``{pipeline_name: {metric: value}}`` metrics per pipeline."""

    dataset: Path | None = None
    """Gold dataset the stage was scored against, if known."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """UTC ISO timestamp of when the stage ran."""

    @property
    def pipelines(self) -> list[str]:
        """Names of the benchmarked pipelines, in run order."""
        return list(self.results)

    def __getitem__(self, pipeline: str) -> dict[str, Any]:
        """Metrics for one pipeline (``result["minhash"]``)."""
        return self.results[pipeline]

    def headline(self, pipeline: str) -> float | None:
        """Value of the stage's headline metric (e.g. F1) for *pipeline*."""
        value = self.results[pipeline].get(STAGE_META[self.stage]["headline"])
        return float(value) if isinstance(value, int | float) else None

    def best_pipeline(self) -> str:
        """Name of the pipeline with the best headline metric."""
        return max(self.pipelines, key=lambda p: self.headline(p) or float("-inf"))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """The plain ``{pipeline: metrics}`` dict (JSON-serializable)."""
        return self.results

    def __repr__(self) -> str:
        return f"StageResult(stage={self.stage!r}, pipelines={self.pipelines!r})"

    def __str__(self) -> str:
        # Blank lines before/after so printing several results doesn't run
        # the tables together.
        return f"\n{format_stage(self.stage, self.results, dataset=self.dataset)}\n"


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
