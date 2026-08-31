# KGLab

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

KGLab is a Python library for research in knowledge graph (KG) generation from raw text, to production-ready KGs for LLM training, graph-based retrieval, and other downstream KG related applications.

> The README is kept intentionally high-level; for detailed walkthroughs, see the [tutorials](tutorial/).

It is designed to support:

- modular KG generation pipelines: swap chunking, extraction, resolution, and export stages without rewriting the pipeline (see Tutorial: [create_custom_pipeline.ipynb](tutorial/create_custom_pipeline.ipynb))
- storage-agnostic execution: run locally or stream directly into disk-backend (e.g. Neo4j) (helpful when building large KGs) without changing the core graph-building logic (see Tutorial: [kg_storage_agnostic.ipynb](tutorial/kg_storage_agnostic.ipynb))
- reproducible benchmarking: compare KG pipeline variants under a shared evaluation setup (see Tutorial: [benchmarking.ipynb](tutorial/benchmarking.ipynb))
- production-ready KG generation pipelines for LLM training and RAG (see Tutorial: [kg_for_llm_training.ipynb](tutorial/kg_for_llm_training.ipynb))

<details>
<summary><strong> README contents</strong></summary>

- [KGLab](#kglab)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [About the Project](#about-the-project)
  - [Project layout](#project-layout)
  - [Contributing](#contributing)
  - [License](#license)

</details>


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

The simplest workflow is to download a small Wikipedia sample and then run a built-in pipeline on it:

```python
from kglab.data import Data, RandomSampler
from kglab.pipelines import Baseline

Data.download(
    "wikipedia",                     # dataset
    path="data/wikipedia/",          # write to path
    sampler=RandomSampler(count=10), # uniformly random sample 10 wiki articles
)

# Builds the knowledge graph and writes the resulting artifact files to output_dir
pipe = Baseline(
    input_paths=["data/wikipedia/"],
    output_dir="output/baseline/",
)

pipe.execute()
```

This runs the full pipeline lifecycle and writes graph artifacts to the output directory.



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
