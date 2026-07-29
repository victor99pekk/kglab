# Polygraph: KG-Grounded SFT Data for LLMs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🌐 Find raw documents → 🧠 Build knowledge graph → 🎯 Train LLM

Research toolkit for building high-quality knowledge graphs and using them to ground LLM training data.

Research toolkot for building highly customizable Knowledge-Graph generation pipelines. This repo provides support to use built KG-generation pipelines, and to customize them by overriding pipeline stages, such as preprocessing stages (chunking, cleaning, deduping, etc...), as well as knowledge building stages like extraction- or resolvation- of entities. The pipelines are implemented as classes that can be easily benchmakred with pre-defined code. The hope is that this will make it easy for people to use existing pipelines (defined in this repo), modifying them, and benchmarking the change with minimal effort and code. This repo also contains support for training GNNs to enhance knowledge graphs.

<details>
<summary><strong>📑 Contents</strong></summary>

- [Polygraph: KG-Grounded SFT Data for LLMs](#polygraph-kg-grounded-sft-data-for-llms)
  - [About the Project](#about-the-project)
  - [Creating a Pipeline Variant](#creating-a-pipeline-variant)
  - [Benchmarking \& Experiments](#benchmarking--experiments)
    - [Neo4j export](#neo4j-export)
  - [ML Model Training](#ml-model-training)
  - [Quick Start](#quick-start)
    - [Hackathon Results](#hackathon-results)
  - [Adding a new build\_kg method](#adding-a-new-build_kg-method)
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
# src/polygraph/pipelines/my_variant.py
from polygraph.pipelines import Baseline
from polygraph.kg_build import extract, resolve, build

class MyVariant(Baseline):
    """Same as Baseline but with custom build_kg."""

    def build_kg(self, chunks):
        entities, triples = extract.with_methods(
            chunks, self.ontology,
            entity_method="spacy",
            relation_method="structured_llm",
        )
        resolved = resolve.by_embedding(entities, threshold=0.85)
        graph = build.from_resolved(resolved, triples)
        return {"graph": graph, "entities": resolved, "triples": triples}
```

Register in `src/polygraph/pipelines/__init__.py`:

```python
from polygraph.pipelines.my_variant import MyVariant
PIPELINE_REGISTRY["my_variant"] = MyVariant
```

## Benchmarking & Experiments

All pipelines can be run and compared via experiment YAML configs:

```bash
# Run a pipeline experiment — produces results_summary.json with all metrics
make experiment EXP=kg/001_baseline
```

```yaml
# experiments/kg/NNN_my_experiment/config.yaml
name: "My experiment"
pipeline:
  variant: "baseline"          # picks from PIPELINE_REGISTRY
input:
  paths:
    - "data/wikipedia/"
output:
  dir: "outputs/"              # generated KGs, metrics, reports
ontology: "configs/default_ontology.yaml"
neo4j:
  upload: true                 # export to Neo4j (requires env vars)
  clear: false
```

The `BenchmarkRunner` runs the pipeline, collects metrics, exports the KG, and
writes a `results_summary.json` — making experiments reproducible and comparable.
See `experiments/kg/_template/` for a full annotated config.

### Neo4j export

Set `neo4j.upload: true` in the experiment config (or pass `--neo4j` to `main.py`).
Requires `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` environment variables.

## ML Model Training

Train models to improve KG quality — entity resolution, node classification,
link prediction. Training code lives in `src/ml/`; inference tools are in
`src/polygraph/models/`.

```bash
cp -r experiments/ML_models/_template/ experiments/ML_models/001_my_experiment/
# edit config.yaml → set task, model variant, hyperparameters
```

Multiple architecture variants per task, selected by YAML config:

```yaml
training:
  task: "node_classification"
  model:
    variant: "gcn"      # or "gat" — picks from MODEL_REGISTRY
  epochs: 100
```

GNN models use PyTorch Geometric — install with `uv sync --extra gnn`.
See `src/ml/README.md` for the full guide on adding new tasks.

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

The KG pipeline is three swappable stages: **extract → resolve → build**.
Each uses a lazy registry (`registry.py`) mapping names to implementations.

**1. Drop a backend** — subclass `EntityExtractor`, `RelationExtractorMethod`,
or follow the resolve/build signatures in CONTRIBUTING.md.

**2. Register it** in the corresponding `registry.py`:
```python
ENTITY_METHODS["my_extractor"] = (
    "polygraph.kg_build.extract.entity.my_extractor:MyExtractor"
)
```

**3. Use it** by name in a pipeline variant or experiment config.

See [CONTRIBUTING.md](CONTRIBUTING.md) for backend signatures and a full walkthrough.

## Architecture

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


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
