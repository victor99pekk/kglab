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
  - [Quick Start](#quick-start-1)
    - [Hackathon Results](#hackathon-results)
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
        entities, triples = extract.with_graphgen(chunks, model="deepseek-v4-pro")
        resolved = resolve.by_embedding(entities, threshold=0.85)
        graph = build.from_resolved(resolved, triples)
        return {"graph": graph, "entities": resolved, "triples": triples}
```

## Quick Start

## Quick Start

```bash
uv sync                                    # install dependencies
python -m spacy download en_core_web_sm    # download NER model

# Run the baseline pipeline
uv run python main.py

# Custom input / output
uv run python main.py -i data/my_corpus/ -o output/experiment_1/

# Run tests
uv run pytest
```

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

## Architecture

```
src/polygraph/
├── pipelines/        # swappable pipeline variants (baseline.py)
├── preprocess/       # load → clean → chunk → quality → dedup
├── kg_build/         # extract entities → resolve → build graph
├── kg_eval/          # quality metrics + structural audit
├── kg_export/        # JSON, GraphML, Neo4j
├── finetune/         # generate training data from KG
└── _shared/          # config, identity, types
```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
