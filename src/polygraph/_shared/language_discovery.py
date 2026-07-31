"""Language discovery from folder structure.

Pipeline ``supported_languages`` is derived from the presence of
2-letter ISO 639-1 subdirectories in each stage folder.  Example
structure::

    clean/
      en/            ← English supported
        normalizer.py
      fr/            ← French supported
        normalizer.py

Stages without language subdirectories (flat files only) are treated
as universal (all languages supported).

Provides:
    * ``discover_stage_languages`` — scan one stage folder
    * ``discover_pipeline_languages`` — intersect across multiple stages
"""

from __future__ import annotations

from pathlib import Path


def discover_stage_languages(stage_path: Path) -> set[str]:
    """Return the set of language codes supported by a stage folder.

    Scans *stage_path* for subdirectories whose names are exactly
    2 lowercase ASCII letters (ISO 639-1 codes).  If no such
    directories exist, returns ``{"*"}`` (universal).

    Args:
        stage_path: Path to a stage folder, e.g.
            ``Path("preprocess/clean")``.

    Returns:
        Set of language codes, or ``{"*"}`` if the stage has
        no per-language subdirectories.
    """
    if not stage_path.is_dir():
        return {"*"}

    langs: set[str] = set()
    for entry in stage_path.iterdir():
        if entry.is_dir() and len(entry.name) == 2 and entry.name.isalpha():
            langs.add(entry.name.lower())
    return langs or {"*"}


def discover_pipeline_languages(*stage_paths: Path) -> set[str]:
    """Compute the intersection of ``discover_stage_languages`` across stages.

    Args:
        *stage_paths: One or more stage folder paths.

    Returns:
        Intersection of all language sets.  Universal stages (``"*"``)
        are excluded from the intersection.  Returns ``set()`` if
        no language-constrained stages exist.
    """
    result: set[str] | None = None
    for path in stage_paths:
        langs = discover_stage_languages(path)
        if "*" in langs:
            continue  # universal stage — doesn't constrain
        if result is None:
            result = set(langs)
        else:
            result &= langs
    return result or set()
