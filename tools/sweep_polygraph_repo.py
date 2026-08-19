#!/usr/bin/env python3
"""Sweep repository (excluding docs/output/.venv/generated_KGs/.git) and replace
`polygraph` -> `kglab`, `Polygraph` -> `KGLab` in all files (including notebooks).
Backs up changed files as <file>.bak.
"""

import re
from pathlib import Path

ROOT = Path(".").resolve()
EXCLUDE_DIRS = {".venv", "output", "generated_KGs", ".git", "docs"}
PAT_LOWER = re.compile(r"\bpolygraph\b")
PAT_TITLE = re.compile(r"\bPolygraph\b")

changed = 0
for p in ROOT.rglob("*"):
    parts = set(p.parts)
    if parts & EXCLUDE_DIRS:
        continue
    if not p.is_file():
        continue
    if p.name.endswith(".bak"):
        continue
    try:
        text = p.read_text(encoding="utf8")
    except Exception:
        continue
    new = PAT_LOWER.sub("kglab", text)
    new = PAT_TITLE.sub("KGLab", new)
    if new != text:
        bak = p.with_suffix(p.suffix + ".bak")
        bak.write_text(text, encoding="utf8")
        p.write_text(new, encoding="utf8")
        print("Updated", p)
        changed += 1

print("Done. Files updated:", changed)
