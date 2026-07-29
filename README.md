# Polygraph: KG-Grounded SFT Data for LLMs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🌐 Find raw documents → 🧠 Build knowledge graph → 🎯 Train LLM

A research toolkit for building highly customizable Knowledge-Graph generation pipelines. This repo provides support for using pre-built KG-generation pipelines and for customizing them by overriding pipeline stages, such as preprocessing stages (chunking, cleaning, deduping, etc.), as well as knowledge-building stages like entity extraction and resolution. The pipelines are implemented as classes that can be easily benchmarked with pre-defined code. The hope is that this will make it easy for people to use the existing pipelines defined in this repo, modify them, and benchmark the changes with minimal effort and code. This repo also contains support for training GNNs to enhance knowledge graphs.

<details>
<summary><strong>📑 Contents</strong></summary>

- [Polygraph: KG-Grounded SFT Data for LLMs](#polygraph-kg-grounded-sft-data-for-llms)
  - [About the Project](#about-the-project)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Build a Knowledge Graph](#build-a-knowledge-graph)
    - [Create new KG-generation pipelines](#create-new-kg-generation-pipelines)
    - [Upload KG to Neo4j](#upload-kg-to-neo4j)
  - [Contributing](#contributing)
  - [License](#license)

</details>

## About the Project

<img src="figures/meta_award.png" alt="Meta Award" width="350" align="right"/>

Polygraph began as a hackathon project at the **Vietnam AI Innovation Challenge**, co-organized by the National Innovation Center (NIC), Meta, and the AI for Vietnam Foundation. Built over 48 hours, it tackled the real-world problem of generating high-quality, fact-grounded training data for LLMs.

The project won the **$5,000 USD Meta Prize** and has since been refactored into a modular research toolkit for studying how knowledge graph quality affects downstream LLM performance.

<details>
<summary><strong>🏆 Hackathon Results</strong></summary>

During the 48-hour competition, we ran an ablation study on 10 Wikipedia articles with 50 held-out test samples:

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

### Build a Knowledge Graph

```python
from polygraph.pipelines import Baseline

pipeline = Baseline(
    input_paths=["data/my_corpus/"],
    output_dir="output/my_experiment/",
)
pipeline.execute()
```

The KG is written to `output/my_experiment/` as `knowledge_graph.json` and
`knowledge_graph.graphml`, plus a `metrics.json` with quality scores.

### Create new KG-generation pipelines
Subclass a pipeline, override stages, and compare against the baseline (e.g., new extraction and resolution methods in the KG build stage):

Custom methods like `with_new_extraction_method` and `with_new_resolve_method` are wired in `kg_build/__init__.py` — add your extraction and resolution backends there so they're callable as `extract.with_new_method` and `resolve.with_new_method`. Register it in `src/polygraph/pipelines/__init__.py`:

```python
# src/polygraph/pipelines/my_variant.py
from polygraph.pipelines import Baseline
from polygraph.kg_build import extract, resolve, build

class MyVariant(Baseline):
    def build_kg(self, chunks):
        entities, triples = extract.with_new_extraction_method(
            chunks, self.ontology,
            entity_method="spacy",
            relation_method="structured_llm",
        )
        resolved = resolve.with_new_resolve_method(entities, threshold=0.85)
        graph = build.from_resolved(resolved, triples)
        return {"graph": graph, "entities": resolved, "triples": triples}
```

```python
from polygraph.pipelines.my_variant import MyVariant
PIPELINE_REGISTRY["my_variant"] = MyVariant
```

Then instantiate and run your pipeline:

```python
pipeline = MyVariant(
    input_paths=["data/wikipedia/"],
    output_dir="experiments/kg/002_my_variant/outputs/",
)
pipeline.execute()
```

### Upload KG to Neo4j

```python
from polygraph.kg_export.neo4j.upload import upload_graph

upload_graph(
    "output/my_experiment/knowledge_graph.json",
    clear=True,
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password",
)
```



<!--
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
See `src/ml/README.md` for the full guide on adding new tasks. -->

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
