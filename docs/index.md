# Polygraph

Research toolkit for building high-quality knowledge graphs for LLM training.

---

## Quick Start

```bash
git clone git@github.com:victor99pekk/polygraph.git
cd polygraph
uv sync
```

```python
from polygraph.pipelines import Baseline

pipe = Baseline(input_paths=["data/"], output_dir="output/")
pipe.execute()
```

---

## Pipeline Stages

1. **Ingest** — Load from JSONL, TXT, JSON, CSV
2. **Preprocess** — Clean, dedup, chunk, quality filter
3. **Extract** — NER + relation extraction → triples
4. **Resolve** — Merge duplicate entities
5. **Build & Export** — Graph construction, JSON/GraphML/Neo4j export

---

## Pages

- [Tutorial](docs/tutorial.md) — Step-by-step walkthrough
- [API Reference](docs/api_reference.md) — All public classes and functions
- [Input Data Format](docs/input_data_format.md) — JSONL schema
- [Contributing](CONTRIBUTING.md) — How to add custom pipelines
