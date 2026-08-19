#!/usr/bin/env python3
"""Find-and-replace `polygraph` -> `kglab` across the repo for code, tests, and docs.

Rules:
- Only replace word-boundary occurrences (regex) to avoid changing paths inside outputs or binary files.
- Skip directories: .venv, output, generated_KGs, src/*egg-info, src/kglab.egg-info, .git
- Changes are made in-place.
- Creates a .rename_backup file per modified file with the original content.

Run: python tools/rename_polygraph_to_kglab.py
"""

import re
from pathlib import Path

ROOT = Path(".").resolve()
EXCLUDE_DIRS = {
    ".venv",
    "output",
    "generated_KGs",
    "src/kg_generator.egg-info",
    "src/kglab.egg-info",
    ".git",
    "venv",
    "__pycache__",
}
FILE_PATTERNS = ["*.py", "*.md", "*.rst", "*.toml", "*.yaml", "*.yml", "*.ini"]

pattern = re.compile(r"\bpolygraph\b")

count = 0
for p in ROOT.rglob("*"):
    if any(part in EXCLUDE_DIRS for part in p.parts):
        continue
    if p.is_file() and any(p.match("**/" + pat) for pat in FILE_PATTERNS):
        text = p.read_text(encoding="utf8")
        if "kglab" in text:
            new = pattern.sub("kglab", text)
            if new != text:
                bak = p.with_suffix(p.suffix + ".rename_backup")
                bak.write_text(text, encoding="utf8")
                p.write_text(new, encoding="utf8")
                print("Updated", p)
                count += 1

print("Done. Files updated:", count)
