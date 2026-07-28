# Polygraph: KG-Grounded SFT Data for LLMs

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🌐 Find raw documents → 🧠 Build knowledge graph → 💬 Generate QA pairs → 🎯 Fine-tune LLM

Research toolkit for building high-quality knowledge graphs and using them to ground LLM training data.

<details>
<summary><strong>📑 Contents</strong></summary>

- [Polygraph: KG-Grounded SFT Data for LLMs](#polygraph-kg-grounded-sft-data-for-llms)
  - [About the Project](#about-the-project)
  - [Creating a Pipeline Variant](#creating-a-pipeline-variant)
  - [Quick Start](#quick-start)
    - [Hackathon Results](#hackathon-results)
  - [Adding a new build_kg method](#adding-a-new-build_kg-method)
  - [Architecture](#architecture)
  - [Contributing](#contributing)
  - [License](#license)

</details>

## About the Project

<img src="figures/meta_award.png" alt="Meta Award" width="350" align="right"/>

Polygraph began as a hackathon project at the **Vietnam AI Innovation Challenge** co-organized by the National Innovation Center (NIC), Meta, and the AI for Vietnam Foundation. Built over 48 hours, it took on the real-world problem of generating high-quality, fact-grounded training data for LLMs.

The project won the **$5,000 USD Meta Prize** and has since been refactored into a modular research toolkit for studying how knowledge graph quality affects downstream LLM performance.

## Creating a Pipeline Variant

```python
# src/polygraph/pipelines/graphgen.py
from polygraph.pipelines import Baseline
from polygraph.kg_build import extract, resolve, build

class GraphGenVariant(Baseline):
    """Same as Baseline but uses LLM extraction."""

    def build_kg(self, chunks):
        entities, triples = extract.graphgen(chunks, model="gpt-4o")
        resolved = resolve.embedding(entities, threshold=0.85)
        graph = build.default(resolved, triples)
        return {"graph": graph, "entities": resolved, "triples": triples}
```

## Quick Start

```bash
# One-time setup
make install

# Verify everything works
make test

# Build a knowledge graph (preprocess → build KG → evaluate → export)
make build-kg

# Custom input / output
make build-kg INPUT=data/my_corpus/ OUTPUT=output/experiment_1/

# Or run directly with uv / python:
uv run python main.py -i data/my_corpus/ -o output/experiment_1/
```

You can also import and run the pipeline programmatically:

```python
from polygraph.pipelines import Baseline

pipeline = Baseline(
    input_paths=["data/my_corpus/"],
    output_dir="output/experiment_1/",
)
pipeline.execute()
```

See `make help` for all available targets.

**Outputs** (in `output/baseline/`): `knowledge_graph.json`, `knowledge_graph.graphml`, `metrics.json`

### Hackathon Results

During the 48-hour competition, we ran an ablation study on 10 Wikipedia articles with 50 held-out test samples:

| Metric | Base Model | KG-Trained (B) | Flat (C) | Improvement |
|---|---|---|---|---|
| **Factual Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Multi-hop Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Hallucination Rate** | 94% | **0%** | 36% | Eliminated entirely |
| **Consistency Score** | 0.56 | **0.83** | 0.76 | +48% |

> *(B) KG-Trained: fine-tuned on QA pairs from the knowledge graph. (C) Flat: QA pairs from the same documents without KG structuring.*

These early results suggested that KG-structured training data could eliminate hallucinations and deliver 6.8× better factual accuracy, motivating further development into a general research toolkit.

## Adding a new build_kg method

The KG pipeline is three swappable stages: **extract → resolve → build**. Each lives in a folder under `kg_build/` with one `.py` file per method.

**1. Drop a backend file** — e.g. `kg_build/extract/my_method.py`:
```python
def my_method(chunks, **kwargs):
    entities = [...]   # your custom extraction logic
    triples = [...]    # (subject_id, predicate, object_id, evidence, chunk_id)
    return entities, triples
```

**2. Wire it** in `kg_build/__init__.py` (2 lines):
```python
from polygraph.kg_build.extract.my_method import my_method
extract.my_method = my_method
```

**3. Use it** in a pipeline variant — then run `python main.py --variant myvariant`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for backend signatures and a full walkthrough.

## Architecture

```
src/polygraph/
├── pipelines/        # swappable variants — subclass Baseline, override stages
├── preprocess/       # load/  clean/  chunk/  quality/  dedup/
├── kg_build/         # extract/  resolve/  build/
├── kg_eval/          # metrics/  structural/
├── kg_export/        # json/  graphml/  neo4j/  rdf/
├── finetune/         # dataset/
└── _shared/          # config, identity, types
```

Each stage folder holds one `.py` file per method. Add a file to add a method.


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
