#!/usr/bin/env python3
"""Replace polygraph -> kglab and Polygraph -> KGLab in docs/ and output/.
Backs up each changed file as <file>.bak.
"""

import re
from pathlib import Path

ROOT = Path(".")
TARGET_DIRS = ["docs", "output"]
PAT_LOWER = re.compile(r"\bpolygraph\b")
PAT_TITLE = re.compile(r"\bPolygraph\b")

changed = 0
for d in TARGET_DIRS:
    for p in (ROOT / d).rglob("*"):
        if not p.is_file():
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
