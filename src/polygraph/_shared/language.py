"""Language-aware pipeline support — stage declarations and skip tracking.

Provides:
    * ``ALL_LANGUAGES`` — sentinel meaning "this stage works for any language"
    * ``filter_by_language`` — split docs into supported and skipped
    * ``SkippedDocs`` — collects and reports skipped documents
    * ``get_language_support`` — introspect a stage's supported languages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Sentinel value — stage supports all languages.
ALL_LANGUAGES = "*"


@dataclass
class SkippedDocs:
    """Tracks documents skipped by pipeline stages due to language mismatch.

    Usage::

        skipped = SkippedDocs()
        kept, s = filter_by_language(docs, {"en"})
        skipped.add("clean", s)
        print(skipped.summary())
    """

    by_stage: dict[str, list[str]] = field(default_factory=dict)

    def add(self, stage: str, docs: list[Any]) -> None:
        """Record *docs* as skipped by *stage*."""
        doc_ids = [d.doc_id for d in docs]
        if doc_ids:
            if stage not in self.by_stage:
                self.by_stage[stage] = []
            self.by_stage[stage].extend(doc_ids)

    @property
    def total(self) -> int:
        """Total number of document-stage skips."""
        return sum(len(v) for v in self.by_stage.values())

    def summary(self) -> str:
        """Human-readable report of all skipped documents."""
        if not self.by_stage:
            return "[language] No documents skipped."
        lines = [f"[language] {self.total} document-stage skips:"]
        for stage, doc_ids in self.by_stage.items():
            preview = doc_ids[:5]
            suffix = "..." if len(doc_ids) > 5 else ""
            lines.append(f"  {stage}: {len(doc_ids)} docs — {preview}{suffix}")
        return "\n".join(lines)


def filter_by_language(
    docs: list[Any],
    supported: set[str],
) -> tuple[list[Any], list[Any]]:
    """Split *docs* into (kept, skipped) based on language support.

    Args:
        docs: Documents to filter.
        supported: Set of language codes the stage supports, or ``{"*"}``
            for universal support.

    Returns:
        ``(kept, skipped)`` tuple.
    """
    if ALL_LANGUAGES in supported:
        return docs, []
    kept = [d for d in docs if d.language in supported]
    skipped = [d for d in docs if d.language not in supported]
    return kept, skipped


def get_language_support(stage: Any) -> set[str]:
    """Introspect a stage's ``supported_languages`` attribute.

    Returns ``{"*"}`` if the attribute is missing (assumes universal).
    """
    return getattr(stage, "supported_languages", {ALL_LANGUAGES})
