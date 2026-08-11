# Contributing

Thanks for your interest in contributing to Polygraph!

## Setup

```bash
git clone https://github.com/your-org/polygraph.git
cd polygraph
uv sync
uv run python -m spacy download en_core_web_sm
uv run pytest tests/ -v
```

## Code Style

- Python 3.10+ with type hints
- Format: `ruff format src/ tests/`
- Lint: `ruff check src/ tests/`

## Project Structure

```
src/polygraph/
├── pipelines/       # Swappable pipeline variants (subclass + override stages)
├── preprocess/      # load/ clean/ chunk/ quality/ dedup/  — one folder per stage
├── kg_build/        # extract/ resolve/ build/             — one folder per stage
├── kg_eval/         # metrics/ structural/
├── kg_export/       # json/ graphml/ neo4j/ rdf/
├── finetune/        # dataset/
└── _shared/         # types, config, identity
```

Every stage folder has one `.py` file per method. Add a file to add a method.

## Adding a new `build_kg` method

The KG pipeline is: **chunks → extract → resolve → build → graph**. Each stage is swappable.

### 1. Add a backend (extraction, resolution, or graph construction)

Drop a `.py` file in the right folder:

```
kg_build/extract/my_extractor.py     # (chunks, **kw) → (entities, triples)
kg_build/resolve/my_resolver.py      # (entities, threshold, **kw) → resolved
kg_build/build/my_builder.py         # (resolved, triples, **kw) → graph
```

Wire it in `kg_build/__init__.py` (2 lines):
```python
from polygraph.kg_build.extract.my_extractor import my_extractor
extract.my_extractor = my_extractor
```

### 2. Create a pipeline variant

```python
# pipelines/my_variant.py
from polygraph.pipelines import Baseline
from polygraph.kg_build import build_kg_into, extract, resolve
from polygraph.kg_build.build import NetworkXGraphWriter

class MyVariant(Baseline):
    def build_kg(self, chunks):
        entities, triples = extract.my_extractor(chunks)
        resolved = resolve.by_string(entities, threshold=0.85)
        writer = NetworkXGraphWriter()
        build_kg_into(writer, chunks, resolved, triples)
        return {"graph": writer.graph, "entities": resolved, "triples": triples}
```

Export it in `pipelines/__init__.py`:
```python
from polygraph.pipelines.my_variant import MyVariant
```

Run: `python main.py --variant myvariant`

### Backend signatures

| Stage | Signature | Returns |
|---|---|---|
| `extract` | `(chunks, **kwargs)` | `(list[dict], list[tuple])` |
| `resolve` | `(entities, threshold, **kwargs)` | `list[dict]` |
| `build_kg_into` | `(writer, chunks, resolved, triples)` | any `GraphWriter` backend |

Triples: `(subject_id, predicate, object_id, evidence_text, source_chunk_id)`

## Pull Requests

1. Create a feature branch
2. Add tests
3. Ensure `uv run pytest tests/ -v` passes
4. Submit a PR with a clear description
