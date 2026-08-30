# KGLab

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

KGLab is a Python toolkit for building knowledge graphs from raw text and turning them into useful assets for LLM training, evaluation, and graph retrieval workflows.

It is designed to be:

- modular: swap chunking, extraction, resolution, and export stages without rewriting the pipeline
- benchmarkable: compare pipeline variants under a shared evaluation setup
- storage-agnostic: run locally or stream into Neo4j without changing the graph-building logic
- practical: generate graph assets, QA data, and exported datasets for downstream ML workflows

## About the Project

<img src="figures/meta_award.png" alt="Meta Award" width="350" align="right"/>

KGLab began as a hackathon project at the **Vietnam AI Innovation Challenge** where we tackled the problem of creating a data management system for LLM training on vietnamese data. We built a pipeline that scrapes the web for vietnamesse text data, cleans it, generates a knowledge graph, and fine-tunes an LLM on it.

The project won the **$5,000 USD Meta Prize** and has since been refactored into a research project for studying how knowledge graphs

<details>
<summary><strong>More about hackathon...</strong></summary>

During the competition we tackled the goal of data management in LLM training on vietnamese data, where the task was to build a data management system. We chose to approach this problem by building a pipeline that scraped vietnamese websites for text data, and from this generates knowledge graph (insert why knowledge graph). We then wanted to prove that our knowledge graph enriched the training data. We did this by doing fine tuning an LLM (Qwen2.5-1.5B) on vietnamese text data. We compared three variants, (i) the original LLM-model, (ii) the LLM fine tuned on the vietnamese text data, not structured as a knowledge graph (we call this Flat below), and lastly (iii) the LLM fine tuned on the knowledge graph data. It was a small model fine tuned on a small set of text, but we saw that the variant fine tuned on the KG-data performed by far best in later benchmarking. Worth noting is that though these benchmarks show promise, it is far from enough to ensure a trend as the training data was too little as we didnt have time to perform better test duringt the 48 hours.

| Metric | Base Model | KG-Trained (B) | Flat (C) | Improvement |
|---|---|---|---|---|
| **Factual Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Multi-hop Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Hallucination Rate** | 94% | **0%** | 36% | Eliminated entirely |
| **Consistency Score** | 0.56 | **0.83** | 0.76 | +48% |

> *(B) KG-Trained: fine-tuned on QA pairs from the knowledge graph. (C) Flat: QA pairs from the same documents without KG structuring.*

These early results suggested that KG-structured training data could eliminate hallucinations and deliver 6.8× better factual accuracy, motivating further development into a general research toolkit.

</details>

## Why KGLab

KGLab focuses on the full lifecycle of a KG workflow:

1. ingest and normalize documents
2. clean, deduplicate, and chunk corpora
3. extract entities and relations
4. resolve duplicate mentions and entities
5. build a graph and validate it against an ontology
6. export it to JSON, GraphML, RDF, or Neo4j
7. benchmark variants and prepare downstream training data

This is useful for research projects that need reproducible pipeline experimentation, as well as applied work that needs a graph for LLM training or graph-based retrieval.

## Installation

KGLab supports Python 3.10+ and uses uv for development installs.

```bash
git clone https://github.com/victor99pekk/kglab.git
cd kglab
uv venv
source .venv/bin/activate
uv pip install -e "."
```

Useful extras:

```bash
uv pip install -e ".[embeddings]"    # semantic entity resolution
uv pip install -e ".[llm]"           # LLM-based extraction
uv pip install -e ".[neo4j]"         # Neo4j export/upload support
uv pip install -e ".[curation]"      # corpus curation and audit tooling
```

## Quick start

The simplest workflow is to run a built-in pipeline on your corpus:

```python
from kglab.pipelines import Baseline

pipe = Baseline(
    input_paths=["data/my_articles/"],
    output_dir="output/baseline/",
)

pipe.execute()
```

This runs the full pipeline lifecycle and writes graph artifacts to the output directory.

You can also run from the command line:

```bash
kg-gen quick -i data/my_articles/ -o output/baseline
```

Or use a YAML experiment config with the benchmarking runner:

```bash
python main.py --experiment experiments/kg/_template/config.yaml
```

## Current capabilities

### Pipeline architecture

KGLab exposes a modular pipeline interface where each stage can be swapped or customized:

- preprocessing: normalization, deduplication, chunking, quality filtering
- extraction: entity and relation extraction strategies
- resolution: string matching or embedding-based entity merging
- graph build: in-memory or file-backed graph stores
- export: JSON, GraphML, RDF, Cytoscape, Neo4j, and more

### Benchmarking

KGLab includes stage-level and end-to-end benchmarking support via the benchmark pipeline package. This is designed for comparing extraction strategies, resolution methods, chunking strategies, and full KG variants side by side.

```python
from kglab.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

config = ExperimentConfig.from_yaml("experiments/kg/_template/config.yaml")
runner = BenchmarkRunner(config)
runner.run()
```

### Data acquisition and curation

The library includes utilities for:

- downloading public datasets and enrichment sources
- scraping and curating raw corpora for training/evaluation
- generating auditable, deterministic dataset manifests

### LLM and training data workflows

KGLab is oriented toward KG-driven ML workflows, including:

- KG-backed fine-tuning data generation
- graph quality evaluation
- graph RAG and retrieval-oriented graph assets
- graph export for downstream tools and storage backends

## Documentation and tutorials

The detailed step-by-step workflows live in the notebooks and docs, while this README stays focused on the project summary and first-run usage.

- [docs/tutorial.md](docs/tutorial.md) — end-to-end walkthrough
- [docs/usage.md](docs/usage.md) — CLI and setup usage guide
- [docs/api_reference.md](docs/api_reference.md) — public API reference
- [docs/input_data_format.md](docs/input_data_format.md) — JSONL input schema
- [tutorial/create_custom_pipeline.ipynb](tutorial/create_custom_pipeline.ipynb) — build and benchmark a custom pipeline
- [tutorial/benchmarking.ipynb](tutorial/benchmarking.ipynb) — compare pipeline variants
- [tutorial/kg_storage_agnostic.ipynb](tutorial/kg_storage_agnostic.ipynb) — same pipeline with different storage backends
- [tutorial/kg_for_llm_training.ipynb](tutorial/kg_for_llm_training.ipynb) — generate LLM training data from a KG

## Project layout

```text
kglab/                 # core library
├── benchmark_pipeline/ # benchmark runners and result reporting
├── data/               # dataset download and curation helpers
├── kg_build/           # extraction, resolution, graph construction
├── kg_eval/            # evaluation and validation utilities
├── kg_export/          # export backends
├── pipelines/          # baseline and custom pipeline variants
├── preprocess/         # cleaning, chunking, deduplication, quality filtering
├── _shared/            # config and shared types
└── finetune/           # KG-to-training-data utilities

ml/                    # ML experiments and training code
experiments/           # reproducible KG and ML experiment configs
configs/               # ontology and configuration defaults
output/                # generated graph outputs and artifacts
```

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style expectations, and contribution workflow.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
