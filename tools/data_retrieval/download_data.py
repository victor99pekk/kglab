#!/usr/bin/env python3
"""Download Wikipedia data for Polygraph — edit the values below to change behavior."""

from polygraph.data import Data

# ── Edit these ────────────────────────────────────────────────
PATH = "data/wikipedia/connected.jsonl"
COUNT = 2
STRATEGY = "degree"  # "random", "degree", or "specific"
LANGUAGE = "en"
TARGET_DEGREE = 1.0
SNAPSHOT = "20231101"
MAX_SCAN = 10000
ENRICH = True  # safe with degree — skips if links already exist
FORCE = True  # re-download even if file exists
# ──────────────────────────────────────────────────────────────

result = Data.download(
    "wikipedia_random",
    path=PATH,
    count=COUNT,
    strategy=STRATEGY,
    language=LANGUAGE,
    target_degree=TARGET_DEGREE,
    snapshot=SNAPSHOT,
    max_scan=MAX_SCAN,
    enrich=ENRICH,
    force=FORCE,
)
print(f"Done: {result}")
