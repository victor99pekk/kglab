"""Human-readable rendering of benchmark results.

Turns the ``{pipeline: {metric: value}}`` dicts returned by the stage
runners into aligned tables with a headline metric, a best-pipeline
marker, and a plain-English interpretation, so a run reads like a
report instead of a dict dump::

    from polygraph.benchmark_pipeline.render import format_stages

    results = Benchmark.Dedup(dataset=...).run(pipelines={...})
    print(format_stages({"dedup": results}))
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Presentation metadata per stage.  "headline" is the single number to
# compare pipelines on; "columns" is the display order for metric keys
# found in the results; "meaning" is the plain-English interpretation
# shown under the table.
STAGE_META: dict[str, dict[str, Any]] = {
    "dedup": {
        "title": "Dedup — near-duplicate detection (DBLP-ACM gold)",
        "headline": "f1",
        "columns": ["precision", "recall", "f1", "accuracy", "threshold"],
        "meaning": (
            "Higher F1 = better duplicate detection. Case-sensitive methods "
            "score low recall on DBLP-ACM (the duplicates differ in case) — "
            "an honest finding, not a bug."
        ),
    },
    "chunking": {
        "title": "Chunking — chunk boundaries keep gold entities intact",
        "headline": "f1",
        "columns": ["precision", "recall", "f1", "n_samples"],
        "meaning": (
            "Higher F1 = chunks less often cut through a gold entity; "
            "1.0 means every gold entity survives in one chunk."
        ),
    },
    "extraction": {
        "title": "Extraction — NER spans vs gold entities",
        "headline": "f1",
        "columns": ["precision", "recall", "f1", "type_accuracy", "n_scored", "n_samples"],
        "meaning": (
            "Higher F1 = extracted NER spans match the gold annotations. Only "
            "labels in the chosen gold's schema are scored (n_scored shows how "
            "many predictions counted); type_accuracy is how often the type is "
            "also correct. Pick the gold that fits your extractor — "
            "print(Benchmark.Extraction.golds())."
        ),
    },
    "resolution": {
        "title": "Resolution — entity merging (cluster F1)",
        "headline": "f1",
        "columns": ["precision", "recall", "f1", "pairwise_f1", "threshold"],
        "meaning": (
            "Higher F1 = mentions are merged the way the gold clusters group "
            "them. The bundled gold is placeholder T2D data — supply real gold "
            "via dataset= or --dataset for meaningful numbers."
        ),
    },
    "quality": {
        "title": "Quality — keep/reject filter vs gold labels",
        "headline": "accuracy",
        "columns": ["accuracy", "precision", "recall", "f1"],
        "meaning": (
            "Higher accuracy = the filter agrees with the gold keep/reject "
            "labels. Bundled gold is placeholder TACRED data — supply real gold "
            "via dataset= or --dataset."
        ),
    },
    "rag": {
        "title": "RAG — retrieval from the built KG",
        "headline": "f1",
        "columns": [
            "precision",
            "recall",
            "f1",
            "entity_recall_at_k",
            "chunk_recall_at_k",
            "entity_coverage",
        ],
        "meaning": (
            "Higher F1 = the KG retrieves the gold supporting chunks for each "
            "query. Demo queries are drawn from the corpus, so recall is "
            "typically high."
        ),
    },
}


def _fmt(value: float | int) -> str:
    """Format a number for table cells (3 decimals for floats)."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _is_number(cell: str) -> bool:
    """True if *cell* (ignoring a trailing best-marker) parses as a number."""
    try:
        float(cell.removesuffix(" *"))
        return True
    except ValueError:
        return False


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an aligned ASCII table; numeric columns are right-aligned."""
    ncols = len(headers)
    widths: list[int] = []
    aligns: list[str] = []
    for i in range(ncols):
        cells = [row[i] for row in rows]
        aligns.append("r" if all(_is_number(c) for c in cells) else "l")
        widths.append(max(len(headers[i]), *(len(c) for c in cells)))

    sep = "-+-".join("-" * w for w in widths)
    header = " | ".join(
        h.rjust(w) if a == "r" else h.ljust(w)
        for h, a, w in zip(headers, aligns, widths, strict=True)
    )
    lines = [header, sep]
    for row in rows:
        lines.append(
            " | ".join(
                c.rjust(w) if a == "r" else c.ljust(w)
                for c, a, w in zip(row, aligns, widths, strict=True)
            )
        )
    return "\n".join(lines)


def format_stage(stage: str, results: Mapping[str, Mapping[str, Any]]) -> str:
    """Render one stage's results as a readable table.

    Args:
        stage: One of the ``STAGE_META`` keys (``"dedup"``, ``"chunking"``,
            ``"extraction"``, ``"resolution"``, ``"quality"``, ``"rag"``).
        results: ``{pipeline_name: {metric: value}}`` as returned by a
            stage runner.

    Returns:
        A multi-line string: title, aligned table (best headline metric
        marked with ``*``), and a plain-English interpretation.
    """
    meta = STAGE_META[stage]
    if not results:
        return f"{meta['title']}\n  (no results)"

    columns = [c for c in meta["columns"] if any(c in m for m in results.values())]
    if not columns:
        columns = sorted({k for m in results.values() for k in m if k != "runtime_seconds"})
    headline = meta["headline"] if meta["headline"] in columns else columns[0]

    numeric_best = [v for m in results.values() if isinstance((v := m.get(headline)), int | float)]
    best = max(numeric_best) if numeric_best else None

    headers = ["pipeline", *columns, "runtime_s"]
    rows: list[list[str]] = []
    for name, metrics in results.items():
        row = [str(name)]
        for col in columns:
            v = metrics.get(col)
            if isinstance(v, int | float):
                cell = _fmt(v)
                if col == headline and best is not None and v == best:
                    cell += " *"
            else:
                cell = "-" if v is None else str(v)
            row.append(cell)
        rt = metrics.get("runtime_seconds")
        row.append(_fmt(rt) if isinstance(rt, int | float) else "-")
        rows.append(row)

    table = _render_table(headers, rows)
    legend = f"  * = best {headline}"
    return f"{meta['title']}\n{table}\n{legend}\n  {meta['meaning']}"


def format_stages(stage_results: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    """Render several stages (e.g. a full ``tools/run_benchmarks.py`` run).

    Args:
        stage_results: ``{stage_name: {pipeline_name: {metric: value}}}``.

    Returns:
        The tables for all stages, separated by a blank line.
    """
    return "\n\n".join(format_stage(stage, results) for stage, results in stage_results.items())
