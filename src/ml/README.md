# `src/ml/` — ML Training Library

All model architectures, training loops, and data preparation live here —
**parallel** to `src/polygraph/`. The library (`polygraph`) only imports
inference tools from `polygraph.models/`; it never touches training code.

---

## Quick reference

| Concept | Location |
|---|---|
| Training code (models, datasets, loops) | `src/ml/<task>/` |
| Inference tools (what pipelines import) | `src/polygraph/models/<task>.py` |
| Saved checkpoints | `experiments/ML_models/<NNN>_<name>/models/` |
| Experiment configs | `experiments/ML_models/<NNN>_<name>/config.yaml` |
| Shared training utilities | `src/ml/training_utils.py` |
| GNN dependencies | `uv sync --extra gnn` |
| KG pipeline experiments | `experiments/kg/` |
| Benchmark & Neo4j upload | `BenchmarkRunner` in `src/polygraph/benchmark_pipeline/` |

---

## Directory layout

```
src/ml/
├── base_trainer.py               # BaseTrainer ABC — every task subclasses this
├── training_utils.py             # EarlyStopping, MetricTracker, SaveBest, count_parameters
│
├── entity_resolution/            # Task: binary classifier for merging duplicate entities
│   ├── config.py                 #   EntityResolutionConfig dataclass
│   ├── dataset.py                #   EntityPairDataset — positive/negative pairs from KG
│   ├── train.py                  #   EntityResolutionTrainer
│   └── models/                   #   architecture variants
│       ├── __init__.py           #     MODEL_REGISTRY + get_model()
│       ├── mlp.py                #     MLPEntityResolver (skeleton)
│       └── attention.py          #     AttentionEntityResolver (skeleton)
│
└── node_classification/          # Task: GNN for predicting entity categories
    ├── config.py                 #   NodeClassificationConfig dataclass
    ├── dataset.py                #   NodeClassificationDataset — labels from entity.type
    ├── train.py                  #   NodeClassificationTrainer
    └── models/                   #   architecture variants (lazy imports)
        ├── __init__.py           #     MODEL_REGISTRY + get_model()
        ├── gcn.py                #     GCNNodeClassifier (PyTorch Geometric, real)
        └── gat.py                #     GATNodeClassifier (PyTorch Geometric, real)
```

---

## How it works

### The two sides: training vs. inference

```
┌─────────────────────────────────┐    ┌──────────────────────────────┐
│  src/ml/  (training code)       │    │  src/polygraph/models/       │
│                                 │    │  (inference only)            │
│  - model architectures          │    │                              │
│  - dataset preparation          │    │  - ModelRegistry             │
│  - training loops               │    │  - EntityResolutionTool      │
│  - never imported by pipelines  │    │  - NodeClassificationTool    │
│                                 │    │  - what pipelines import     │
└─────────────────────────────────┘    └──────────────────────────────┘
         │                                        ▲
         │  saves checkpoint                      │  loads checkpoint
         ▼                                        │
┌─────────────────────────────────────────────────────────────┐
│  experiments/ML_models/<NNN>_<name>/models/best.pt          │
└─────────────────────────────────────────────────────────────┘
```

### Where training data comes from

Every task reads a **KG JSON file** produced by running a pipeline:

```
Raw text → Pipeline → knowledge_graph.json → ML dataset → Model training
```

The `knowledge_graph.json` contains entities, triples, and graph structure.
Each task's `dataset.py` reads this file and generates task-specific
training samples with labels.

### Pipeline benchmarking & Neo4j export

Pipeline variants in `src/polygraph/pipelines/` can all be run and compared
via `BenchmarkRunner`:

```python
from polygraph.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

config = ExperimentConfig.from_yaml("experiments/kg/001_baseline/config.yaml")
runner = BenchmarkRunner(config)
result = runner.run()   # preprocess → build KG → evaluate → export → results_summary.json
```

Each run produces a `results_summary.json` with all metrics, timestamps, and
config — making experiments reproducible and comparable across pipeline
variants.

Generated KGs can be uploaded to Neo4j by setting `neo4j.upload: true` in
the experiment YAML (requires `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`
env vars).

```yaml
# experiments/kg/001_baseline/config.yaml
neo4j:
  upload: true
  clear: false   # set true to wipe the DB first
```

---

## Adding a new ML task

Follow this checklist. Use `entity_resolution/` and `node_classification/`
as reference implementations.

### Step 1 — Create the task directory

```bash
mkdir -p src/ml/<task_name>/models
```

Create these files:

| File | Purpose |
|---|---|
| `__init__.py` | Public API — exports config, dataset, trainer, model registry |
| `config.py` | `TaskConfig` dataclass — all hyperparameters |
| `dataset.py` | Loads KG JSON, generates labeled samples, splits train/val/test |
| `train.py` | `TaskTrainer(BaseTrainer)` — implements prepare_data, build_model, train, evaluate |
| `models/__init__.py` | `MODEL_REGISTRY` dict + `get_model(variant)` factory |
| `models/<variant>.py` | One file per architecture variant |

### Step 2 — Define the config (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TaskConfig:
    model_variant: str = "default"     # picks from MODEL_REGISTRY
    # ... task-specific hyperparameters ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        return cls(**data)
```

### Step 3 — Define the dataset (`dataset.py`)

The dataset must:

1. Load entities/graph from a KG JSON file
2. Generate labeled samples for the task
3. Provide a `split(train_ratio, val_ratio)` method returning `(train, val, test)`

Labels can come from:
- Entity names (entity resolution — same name = match)
- Entity `type` field (node classification — ontology types)
- Triples (link prediction — existing triples = positive, random = negative)

### Step 4 — Define model variants (`models/<variant>.py`)

Each model must implement:

```python
class MyModel(nn.Module):  # or plain class with save/load
    def forward(self, *inputs) -> Tensor: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "MyModel": ...
```

Register it in `models/__init__.py`:

```python
from ml.<task>.models.variant_a import VariantA
from ml.<task>.models.variant_b import VariantB

MODEL_REGISTRY: dict[str, type] = {
    "variant_a": VariantA,
    "variant_b": VariantB,
}

def get_model(variant: str) -> type: ...
```

### Step 5 — Define the trainer (`train.py`)

Subclass `BaseTrainer` and implement the four abstract methods:

```python
from ml.base_trainer import BaseTrainer
from ml.<task>.models import get_model

class TaskTrainer(BaseTrainer):
    def __init__(self, ...):
        super().__init__(...)
        self.model_config = model_config or TaskConfig()

    def prepare_data(self) -> None:
        """Load KG → generate samples → split train/val/test."""
        ...

    def build_model(self) -> None:
        """Instantiate model via the registry."""
        model_cls = get_model(self.model_config.model_variant)
        self.model = model_cls(**model_kwargs)

    def train(self) -> None:
        """Training loop with EarlyStopping, MetricTracker, SaveBest."""
        ...

    def evaluate(self) -> dict[str, float]:
        """Run test-set evaluation, return metric dict."""
        ...
```

### Step 6 — Package init (`__init__.py`)

```python
from ml.<task>.config import TaskConfig
from ml.<task>.dataset import TaskDataset
from ml.<task>.models import MODEL_REGISTRY, available_variants, get_model
from ml.<task>.train import TaskTrainer

__all__ = ["TaskConfig", "TaskDataset", "TaskTrainer", ...]
```

### Step 7 — Create the inference tool

In `src/polygraph/models/<task_name>.py`:

```python
class TaskTool:
    def load(self, path: str): ...
    def predict(self, *inputs): ...

# Auto-register
from polygraph.models.registry import ModelRegistry
ModelRegistry.register("<task_name>", TaskTool)
```

### Step 8 — Wire up auto-registration

In `src/polygraph/models/__init__.py`, add:

```python
import polygraph.models.<task_name>  # noqa: F401 — registers tool
```

### Step 9 — Create an experiment

```bash
cp -r experiments/ML_models/_template/ experiments/ML_models/<NNN>_<name>/
```

Edit `config.yaml`:

```yaml
name: "NNN — My training experiment"
input:
  kg_path: "experiments/kg/001_baseline/outputs/knowledge_graph.json"
training:
  task: "<task_name>"
  model:
    variant: "default"
  epochs: 100
```

---

## Shared training utilities

All in `src/ml/training_utils.py`:

| Utility | Purpose |
|---|---|
| `EarlyStopping(patience, mode)` | Stop when metric stops improving |
| `MetricTracker(output_dir, metrics)` | Log per-epoch metrics, write `training_log.json` |
| `SaveBest(model_dir, filename, mode)` | Save checkpoint only on improvement |
| `count_parameters(model)` | Count trainable parameters (PyTorch) |

---

## Experiment config reference

Training experiments live in `experiments/ML_models/<NNN>_<name>/`:

```yaml
name: "001 — My experiment"
description: "What this tests"

input:
  kg_path: "experiments/kg/001_baseline/outputs/knowledge_graph.json"

output:
  dir: "outputs/"           # defaults to <experiment_dir>/outputs/

training:
  task: "entity_resolution"  # or "node_classification", "link_prediction"
  epochs: 100
  batch_size: 64
  learning_rate: 0.001

  model:
    variant: "mlp"           # picks from MODEL_REGISTRY
    embedding_dim: 128
    dropout: 0.3

  data:
    min_entity_frequency: 2
    negative_ratio: 3
    train_split: 0.8
    val_split: 0.1
```

Pipeline experiments (KG generation) live in `experiments/kg/`.
