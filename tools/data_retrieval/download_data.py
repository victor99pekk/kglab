#!/usr/bin/env python3
"""Download Wikipedia data for Polygraph — edit the values below to change behavior."""

from polygraph.data import Data, DegreeSampler, RandomSampler, SpecificSampler

# ── Edit these ────────────────────────────────────────────────
PATH = "data/wikipedia/connected.jsonl"
COUNT = 2
STRATEGY = "degree"  # "random", "degree", or "specific"
LANGUAGE = "en"
TARGET_DEGREE = 1.0
SNAPSHOT = "20231101"
MAX_SCAN = 10000
FORCE = True  # re-download even if file exists
# ──────────────────────────────────────────────────────────────

if STRATEGY == "degree":
    sampler = DegreeSampler(count=COUNT, target_degree=TARGET_DEGREE, max_scan=MAX_SCAN)
elif STRATEGY == "specific":
    sampler = SpecificSampler()
else:
    sampler = RandomSampler(count=COUNT, max_scan=MAX_SCAN)

result = Data.download(
    "wikipedia",
    path=PATH,
    sampler=sampler,
    language=LANGUAGE,
    snapshot=SNAPSHOT,
    force=FORCE,
)
print(f"Done: {result}")
