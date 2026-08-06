"""Resolution benchmark — compare entity merging accuracy against gold clusters.

Each pipeline resolves a set of entity names and cluster F1 is computed
against gold entity groups.

Gold dataset format (JSONL)::

    {"names": ["IBM", "International Business Machines", "IBM Corp"], "cluster_id": 1}
    {"names": ["Apple", "Apple Inc"], "cluster_id": 2}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from polygraph._shared.stage_config import ResolutionConfig
from polygraph.benchmark_pipeline.report import StageResult
from polygraph.data import Data
from polygraph.kg_build import resolve
from polygraph.pipelines import Pipeline

#: Default location for the gold resolution dataset when no ``dataset`` is passed.
_DEFAULT_DATASET = "benchmarks/data/resolution_gold.jsonl"


class ResolutionRunner:
    """Benchmark entity resolution quality across pipeline instances.

    Compares how well each pipeline merges entity name variants into
    canonical clusters against gold entity groupings.

    Primary metric: cluster F1 (how well clusters match gold groups).
    """

    def __init__(self, dataset: str | Path = _DEFAULT_DATASET) -> None:
        self.dataset = Path(dataset)

    @classmethod
    def help(cls) -> str:
        """Return a human-readable description of this benchmark.

        Use this to understand what the benchmark measures, the gold
        dataset format, and the metrics it produces — without reading
        the source code.

        Example::

            print(ResolutionRunner.help())
        """
        return (
            "Entity Resolution Benchmark\n"
            "============================\n"
            "What it tests:\n"
            "  Whether a pipeline correctly merges different surface\n"
            "  forms of the same entity (e.g., 'IBM', 'International\n"
            "  Business Machines', 'IBM Corp') into a single canonical\n"
            "  entity. Compares resolved clusters against gold groupings.\n\n"
            "Why it matters:\n"
            "  Unresolved duplicates fragment the KG — the same real-world\n"
            "  entity appears under multiple names, splitting relations\n"
            "  and inflating entity counts. Good resolution produces a\n"
            "  cleaner, more connected graph.\n\n"
            "Gold dataset format (JSONL):\n"
            '  {"names": ["IBM", "International Business Machines", "IBM Corp"],\n'
            '   "cluster_id": 1}\n'
            '  {"names": ["Apple", "Apple Inc"], "cluster_id": 2}\n\n'
            "Primary metrics:\n"
            "  Cluster precision — how pure are the resolved clusters?\n"
            "  Cluster recall — how many gold clusters were recovered?\n"
            "  Cluster F1 — harmonic mean of cluster precision and recall.\n"
            "  Pairwise F1 — F1 over all entity pairs (more granular).\n"
        )

    def run(self, pipelines: dict[str, Pipeline]) -> StageResult:
        """Run resolution benchmark and return metrics per pipeline.

        Args:
            pipelines: ``{name: Pipeline}`` dict.  Each pipeline should be
                fully configured (resolution method, threshold, etc.) but
                does not need ``input_paths`` or ``output_dir``.

        Returns:
            A ``StageResult`` wrapping ``{pipeline_name: {method, threshold,
            precision, recall, f1, pairwise_precision, pairwise_recall,
            pairwise_f1, runtime_seconds}}`` — one entry per pipeline, keyed
            by the name given in ``pipelines``.
        """
        Data.download("bench_resolution", path=str(self.dataset))
        gold = _load_gold(self.dataset)
        names = _unique_names(gold)

        results: dict[str, Any] = {}
        for pipeline_name, pipeline in pipelines.items():
            start = time.monotonic()
            method, threshold, options = _read_resolution_config(pipeline)
            predicted = _resolve_names(names, method, threshold, options)
            metrics = _score(gold, predicted)
            results[pipeline_name] = {
                "method": method,
                "threshold": threshold,
                **metrics,
                "runtime_seconds": time.monotonic() - start,
            }
        return StageResult(stage="resolution", results=results, dataset=self.dataset)


def _load_gold(path: str | Path) -> list[set[str]]:
    """Load gold entity clusters from the resolution JSONL dataset.

    Each record is ``{"names": [...], "cluster_id": ...}``.  Records that
    share a ``cluster_id`` are merged into a single gold cluster.  Names are
    normalized (casefolded + stripped) so they can be compared against the
    resolver output regardless of case.

    Args:
        path: Path to the gold JSONL file.

    Returns:
        List of gold clusters, each a set of normalized entity names.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a record is missing ``names``/``cluster_id`` or has an
            invalid ``names`` value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resolution gold dataset not found: {path}")

    clusters_by_id: dict[Any, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            names = record.get("names")
            cluster_id = record.get("cluster_id")
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(n, str) and n.strip() for n in names)
            ):
                raise ValueError(
                    f"Resolution gold record on line {line_no} has invalid 'names': {record!r}"
                )
            if cluster_id is None:
                raise ValueError(
                    f"Resolution gold record on line {line_no} missing 'cluster_id': {record!r}"
                )
            clusters_by_id.setdefault(cluster_id, set()).update(n.casefold().strip() for n in names)
    return list(clusters_by_id.values())


def _unique_names(gold_clusters: list[set[str]]) -> list[str]:
    """All entity names appearing in any gold cluster (deduplicated).

    These are the surface names handed to each pipeline's resolution stage.

    Args:
        gold_clusters: Gold clusters from :func:`_load_gold`.

    Returns:
        Unique normalized names in first-seen order.
    """
    seen: set[str] = set()
    names: list[str] = []
    for cluster in gold_clusters:
        for name in cluster:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _read_resolution_config(pipeline: Pipeline) -> tuple[str, float, dict[str, Any]]:
    """Read resolution method + threshold from a pipeline's config.

    Prefers the typed ``pipeline.resolution`` attribute (a
    ``ResolutionConfig``), falling back to the raw ``pipeline._config`` dict
    parsed with ``ResolutionConfig.from_dict``, then to defaults.

    Args:
        pipeline: A configured pipeline instance.

    Returns:
        ``(method, threshold, options)`` used by the pipeline's resolution
        stage.
    """
    res_cfg = getattr(pipeline, "resolution", None)
    if isinstance(res_cfg, dict):
        res_cfg = ResolutionConfig.from_dict(res_cfg)
    if res_cfg is None:
        res_cfg = ResolutionConfig.from_dict(pipeline._config.get("resolution"))
    return res_cfg.method, res_cfg.threshold, dict(res_cfg.options)


def _resolve_names(
    names: list[str],
    method: str,
    threshold: float,
    options: dict[str, Any],
) -> list[set[str]]:
    """Resolve entity names into predicted clusters using a resolution method.

    Runs ``polygraph.kg_build.resolve`` with the given method/threshold and
    converts each resolved canonical entity back into a cluster of surface
    names (canonical name + all aliases).

    Args:
        names: Entity surface names to resolve.
        method: Resolution method (e.g., ``"string"``, ``"embedding"``).
        threshold: Similarity threshold for merging.
        options: Extra keyword arguments forwarded to the resolution method.

    Returns:
        List of predicted clusters, each a set of normalized names.

    Raises:
        ValueError: If ``method`` is not a registered resolution method.
    """
    entities = [{"name": name} for name in names]
    resolved = resolve.with_method(entities, method=method, threshold=threshold, **options)

    clusters: list[set[str]] = []
    for canonical in resolved:
        members = {str(canonical.get("name", "")).casefold().strip()}
        for alias in canonical.get("aliases", []):
            members.add(str(alias).casefold().strip())
        members.discard("")
        if members:
            clusters.append(members)
    return clusters


def _score(
    gold_clusters: list[set[str]],
    predicted_clusters: list[set[str]],
) -> dict[str, float]:
    """Compute cluster and pairwise precision/recall/F1.

    Cluster metrics use best-matching between gold and predicted clusters:

    * precision — how pure each predicted cluster is (fraction of members
      belonging to the single best-matching gold cluster).
    * recall — how much of each gold cluster was recovered (fraction of
      members found in the single best-matching predicted cluster).

    Pairwise metrics count entity-name pairs — a pair is a true positive if
    its members share both a gold cluster and a predicted cluster, a false
    positive if they share only a predicted cluster, and a false negative if
    they share only a gold cluster.  Pairwise F1 directly answers "are gold
    cluster members merged together?".

    Args:
        gold_clusters: Gold clusters from :func:`_load_gold`.
        predicted_clusters: Predicted clusters from :func:`_resolve_names`.

    Returns:
        Dict with ``precision``, ``recall``, ``f1`` (cluster-level) and
        ``pairwise_precision``, ``pairwise_recall``, ``pairwise_f1``.
    """
    # name → set of gold cluster indices / predicted cluster index
    gold_membership: dict[str, set[int]] = {}
    for gi, cluster in enumerate(gold_clusters):
        for name in cluster:
            gold_membership.setdefault(name, set()).add(gi)
    pred_membership: dict[str, int] = {}
    for pi, cluster in enumerate(predicted_clusters):
        for name in cluster:
            pred_membership[name] = pi

    # ── Cluster precision / recall (best-match) ────────────────
    total_pred = sum(len(cluster) for cluster in predicted_clusters)
    total_gold = sum(len(cluster) for cluster in gold_clusters)
    pred_best_overlap = sum(
        max((len(pc & gc) for gc in gold_clusters), default=0) for pc in predicted_clusters
    )
    gold_best_overlap = sum(
        max((len(gc & pc) for pc in predicted_clusters), default=0) for gc in gold_clusters
    )
    cluster_precision = pred_best_overlap / total_pred if total_pred else 0.0
    cluster_recall = gold_best_overlap / total_gold if total_gold else 0.0

    # ── Pairwise precision / recall / F1 ───────────────────────
    names = list(gold_membership.keys())
    tp = fp = fn = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            same_gold = bool(gold_membership[a] & gold_membership[b])
            pa = pred_membership.get(a)
            pb = pred_membership.get(b)
            same_pred = pa is not None and pb is not None and pa == pb
            if same_gold and same_pred:
                tp += 1
            elif same_pred and not same_gold:
                fp += 1
            elif same_gold and not same_pred:
                fn += 1
    pairwise_precision = tp / (tp + fp) if (tp + fp) else 0.0
    pairwise_recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "precision": cluster_precision,
        "recall": cluster_recall,
        "f1": _f1(cluster_precision, cluster_recall),
        "pairwise_precision": pairwise_precision,
        "pairwise_recall": pairwise_recall,
        "pairwise_f1": _f1(pairwise_precision, pairwise_recall),
    }


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall.

    Returns 0.0 when both are zero (avoiding division by zero).
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
