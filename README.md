# Polygraph: Customizable Knowledge Graph Pipelines

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<p align="center">
  <img src="figures/graph_readme.png" alt="Polygraph pipeline overview" width="30%"/>
</p>

<p align="right"><sub><small>Image adapted from <a href="https://www.researchgate.net/figure/Left-The-node-link-diagram-view-renders-glyphs-for-nodes-and-curves-for-edges-The-view_fig3_265011275">Holten &amp; van Wijk (2009)</a>.</small></sub></p>

🌐 Find raw documents → 🧠 Build knowledge graph → 🎯 Train LLM / 🔍 Graph RAG

A toolkit for building highly customizable Knowledge-Graph generation pipelines. Our hope with this library is twofold. Each point below is backed by a runnable notebook in [`tutorial/`](tutorial/).

1. `Enable easy research with KG generation.` We want it to be easy to try out different KG generation pipelines. We do this by
    - making it simple to extend existing KG generation pipelines with minimal code — see [create_custom_pipeline.ipynb](tutorial/create_custom_pipeline.ipynb).
    - making the KG generation pipeline modular, enabling highly customizable pipelines built with this library.
    - providing code for benchmarking existing pipelines against new pipelines, with benchmark tests covering chunking, deduplication, extraction, entity resolution, quality, and RAG retrieval — each scored side-by-side — see [benchmarking.ipynb](tutorial/benchmarking.ipynb).

2. `Make it easy to use our built custom KG generation pipelines.` We do this by
    - providing pre-built pipelines that can be customized by setting parameters and customizing specific pipeline stages — see [kg_for_llm_training.ipynb](tutorial/kg_for_llm_training.ipynb) for turning a KG into LLM training data.
    - keeping the graph storage agnostic of the Pipeline classes, so the same pipeline code can be used on small graphs generated locally, or large graphs being generated and continuously streamed to a remote backend (e.g. Neo4j) — see [kg_storage_agnostic.ipynb](tutorial/kg_storage_agnostic.ipynb).

<details>
<summary><strong>📑 Contents</strong></summary>

- [Polygraph: Customizable Knowledge Graph Pipelines](#polygraph-customizable-knowledge-graph-pipelines)
  - [About the Project](#about-the-project)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [End-to-end example](#end-to-end-example)
  - [Tutorial Notebooks](#tutorial-notebooks)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

</details>

## About the Project

<img src="figures/meta_award.png" alt="Meta Award" width="350" align="right"/>

Polygraph began as a hackathon project at the **Vietnam AI Innovation Challenge** where we tackled the problem of creating a data management system for LLM training on vietnamese data. We built a pipeline that scrapes the web for vientamese text data, cleans it, generates a knowledge graph, and fines tunes an LLM on it.

The project won the **$5,000 USD Meta Prize** and has since been refactored into a research project for studying how knowledge graphs

<details>
<summary><strong>More about hackathon...</strong></summary>

During the competition we tackled the goal of data management in LLM training on vietnamese data, where the task was to build a data management system. We chose to approach this problem by building a pipeline that scraped vietnamese websites for text data, and from this generates knowledge graph (insert why knowledge graph). We then wanted to prove that our knowledge graph enriched the training data. We did this by doing fine tuning an LLM (Qwen2.5-1.5B) on vietnamese text data. We compared three variants, (i) the original LLM-model, (ii) the LLM fine tuned on the vietnamese text data, not structured as a knowledge graph (we call this Flat below), and lastly (iii) the LLM fine tuned on the knowledge graph data. It was a small model fine tuned on a small set of text, but we saw that the variant fine tuned on the KG-data performed by far best in later benchmarking. Worth noting is that though these benchmarks show promise, it is far from enough to ensure a trend as the trainnig data was too little as we didnt have time to perform better test duringt the 48 hours.

| Metric | Base Model | KG-Trained (B) | Flat (C) | Improvement |
|---|---|---|---|---|
| **Factual Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Multi-hop Accuracy** | 2.7% | **18.1%** | 4.2% | 6.8× over base |
| **Hallucination Rate** | 94% | **0%** | 36% | Eliminated entirely |
| **Consistency Score** | 0.56 | **0.83** | 0.76 | +48% |

> *(B) KG-Trained: fine-tuned on QA pairs from the knowledge graph. (C) Flat: QA pairs from the same documents without KG structuring.*

These early results suggested that KG-structured training data could eliminate hallucinations and deliver 6.8× better factual accuracy, motivating further development into a general research toolkit.

</details>

## Getting Started

<details>
<summary><strong>📁 Architecture</strong></summary>

```
src/
├── polygraph/           # KG library (core)
│   ├── pipelines/       #   swappable variants — subclass Pipeline
│   ├── benchmark_pipeline/  # BenchmarkRunner — runs any pipeline from YAML config
│   ├── models/          #   inference tools — load trained checkpoints
│   ├── preprocess/      #   load / clean / chunk / quality / dedup
│   ├── kg_build/        #   extract / resolve / build
│   ├── kg_eval/         #   metrics / structural
│   ├── kg_export/       #   json / graphml / neo4j / rdf
│   ├── finetune/        #   QA dataset generation
│   └── _shared/         #   config, identity, types
│
└── ml/                  # ML training (parallel to polygraph)
    ├── base_trainer.py  #   BaseTrainer ABC
    ├── training_utils.py #  EarlyStopping, MetricTracker, SaveBest
    ├── entity_resolution/   # binary classifier for merging entities
    │   └── models/
    ├── node_classification/  # GNN for entity type prediction
    │   └── models/
    └── topic_classification/ # GNN for document topic prediction
        └── models/

experiments/
├── kg/                  # pipeline experiments (config.yaml → BenchmarkRunner)
│   └── _template/
└── ML_models/           # training experiments (config.yaml → BaseTrainer)
    └── _template/
```

</details>


### Installation

**Prerequisites:** Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:victor99pekk/polygraph.git
cd polygraph
make install         # syncs all deps + downloads spaCy model
```

Run `make help` to see all available targets.

### End-to-end example

Create a pipeline variant and benchmark it against the baseline in one test:

```python
from polygraph._shared.stage_config import PreprocessConfig
from polygraph.benchmark_pipeline import Benchmark
from polygraph.pipelines import Baseline, PIPELINE_REGISTRY

# 1. Create a new pipeline — subclass Baseline and override a stage
class SmallChunks(Baseline):
    def __init__(self, **kwargs):
        kwargs.setdefault("preprocess", PreprocessConfig(
            chunk_method="sentence", chunk_target_tokens=300))
        super().__init__(**kwargs)

PIPELINE_REGISTRY["small_chunks"] = SmallChunks  # now discoverable by name

# 2. Benchmark both pipelines — pass them into one benchmark test
result = Benchmark.Dedup(dataset="benchmarks/data/dedup_gold.jsonl").run(
    pipelines={"baseline": Baseline(), "small_chunks": SmallChunks()},
)
print(result)                  # aligned per-pipeline table
print("best:", result.best_pipeline())
```

Every benchmark test (`Dedup`, `Chunking`, `Resolution`, `Extraction`, `Quality`,
`RAG`) accepts any number of pipelines and scores them side by side — that's how
you compare variants, so there is no separate `compare()` API. `sentence` chunking
keeps the demo fast — the default is `semantic`. For a whole-pipeline run
(preprocess → build → export), use `BenchmarkRunner` directly. To fetch data for
such a run, pick a sampler and download:

```python
from polygraph.data import Data, DegreeSampler, RandomSampler, SpecificSampler

Data.download("wikipedia", sampler=RandomSampler(count=20))
# SpecificSampler(urls=[...])                      — fetch explicit articles
# DegreeSampler(count=50, target_degree=5.0)       — grow a connected, link-rich set
```

Downloads are automatically enriched with outgoing Wikipedia hyperlinks.

See [docs/tutorial.md](docs/tutorial.md) for more workflows, including uploading a
KG to Neo4j.

## Tutorial Notebooks

Runnable notebooks in [`tutorial/`](tutorial/) demonstrate the main workflows
end-to-end:

| Notebook | What it covers |
|---|---|
| [create_custom_pipeline.ipynb](tutorial/create_custom_pipeline.ipynb) | Create a new pipeline variant — subclass, register, and benchmark it against the baseline |
| [kg_storage_agnostic.ipynb](tutorial/kg_storage_agnostic.ipynb) | The **same pipeline** built locally and in Neo4j — graph storage is agnostic of the pipeline |
| [kg_neo4j_streaming.ipynb](tutorial/kg_neo4j_streaming.ipynb) | Stream KG creation directly into Neo4j for corpora too large for RAM |
| [benchmarking.ipynb](tutorial/benchmarking.ipynb) | Benchmark pipelines against each other |
| [kg_for_llm_training.ipynb](tutorial/kg_for_llm_training.ipynb) | Turn a generated KG into training data for an LLM |

## Documentation

| Resource | Description |
|---|---|
| [Tutorial](docs/tutorial.md) | Step-by-step walkthrough — from input data to exported KG |
| [Tutorial Notebooks](tutorial/) | Runnable notebooks — custom pipelines, storage-agnostic builds, Neo4j streaming, benchmarking, KG for LLM training |
| [API Reference](docs/api_reference.md) | Complete reference for all public classes and functions |
| [Input Data Format](docs/input_data_format.md) | JSONL schema specification |
| [Contributing Guide](CONTRIBUTING.md) | How to add custom extractors, resolvers, and pipelines |
| [Example Scripts](examples/) | Runnable Python examples for common workflows |


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
