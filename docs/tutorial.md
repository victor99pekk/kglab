# KGLab Tutorial

A step-by-step walkthrough of building, evaluating, and exporting a knowledge
graph with KGLab.

---

## 1. Prepare your input data

KGLab expects **JSONL** input — one JSON object per line, with `id` and
`text` fields (see `docs/input_data_format.md` for the full spec).

```jsonl
{"id": "einstein", "text": "Albert Einstein was a German-born theoretical physicist...", "title": "Albert Einstein", "url": "https://en.wikipedia.org/wiki/Albert_Einstein"}
{"id": "newton", "text": "Sir Isaac Newton was an English mathematician...", "title": "Isaac Newton", "url": "https://en.wikipedia.org/wiki/Isaac_Newton"}
```

Place your `.jsonl` files in a directory (e.g. `data/my_articles/`).

---

## 2. Run a baseline pipeline

The simplest way to build a KG is with the `Baseline` pipeline:

```python
from kglab.pipelines import Baseline

pipe = Baseline(
    input_paths=["data/my_articles/"],
    output_dir="output/baseline/",
)
pipe.execute()
```

This runs the full pipeline: **preprocess → build KG → evaluate → export**.
When it finishes you'll find these files in `output/baseline/`:

| File | Description |
|---|---|
| `knowledge_graph.json` | Full KG with nodes, edges, entities, triples |
| `metrics.json` | Quality evaluation scores |
| `knowledge_graph.graphml` | (optional) GraphML for Gephi/Cytoscape |

---

## 3. Configure extraction

The `Baseline` pipeline accepts typed configuration objects for each stage:

```python
from kglab.pipelines import Baseline
from kglab._shared.stage_config import ExtractionConfig, ResolutionConfig

pipe = Baseline(
    input_paths=["data/my_articles/"],
    output_dir="output/llm_pipeline/",
    extraction=ExtractionConfig(
        mode="composed",
        entity_method="spacy",
        relation_method="structured_llm",
    ),
    resolution=ResolutionConfig(
        method="embedding",
        threshold=0.85,
    ),
)
pipe.execute()
```

**Available extraction modes** (set via `ExtractionConfig.mode`):

| Mode | Description |
|---|---|
| `"composed"` | Separate entity + relation extractors (e.g. spaCy + ontology rules) |
| `"joint"` | Single method that extracts both (e.g. GraphGen LLM) |

**Available entity methods**: `spacy`, `simple`

**Available relation methods**: `ontology_rules`, `structured_llm`

**Available joint methods**: `graphgen`

**Available resolution methods**: `string` (fast, no deps), `embedding` (semantic, needs `sentence-transformers`)

**Available graph backends** (set via `BuildConfig.method`):

| Method | Description |
|---|---|
| `"networkx"` | In-memory graph (default, fast but RAM-bound) |
| `"sqlite"` | File-backed graph (`.db` file, disk-resident, handles large KGs) |

```python
from kglab._shared.stage_config import BuildConfig

# For large graphs — avoids OOM by writing to disk
pipe = Baseline(
    input_paths=["data/"],
    output_dir="output/",
    build=BuildConfig(method="sqlite"),
)
pipe.execute()
```

---

## 4. Evaluate KG quality

You can run evaluation independently of a full pipeline:

```python
from kglab.kg_eval import metrics, structural
from kglab.kg_eval.metrics import AccuracyEvaluator
import json

# Load a previously built KG
with open("output/baseline/knowledge_graph.json") as f:
    kg = json.load(f)

# Basic metrics (no LLM needed)
report = metrics.evaluate(kg["graph"], kg["entities"], kg["triples"])
print(f"Overall score: {report['overall_score']:.2f}")

# Structural audit (check ontology compliance)
audit = structural.run(
    kg["graph"], kg["entities"], kg["triples"],
    ontology_path="configs/default_ontology.yaml",
)

# Accuracy evaluation (requires LLM)
def my_llm(prompt: str) -> str:
    # Replace with your LLM call — any (str) -> str callable works
    import openai
    return openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

accuracy_eval = AccuracyEvaluator(llm_client=my_llm)
accuracy_report = accuracy_eval.evaluate(kg["graph"], kg["entities"], kg["triples"])
```

---

## 5. Export to other formats

Export a KG to various formats programmatically:

```python
from kglab.kg_export import GraphExporter
from pathlib import Path

exporter = GraphExporter()
paths = exporter.export(
    graph=kg["graph"],
    entities=kg["entities"],
    triples=kg["triples"],
    output_dir=Path("output/exports/"),
    formats=["json", "graphml", "neo4j_csv", "rdf", "cytoscape"],
)
for p in paths:
    print(f"  → {p}")
```

Or use the convenience function API:

```python
from kglab.kg_export import exporter

exporter.to_json(kg["graph"], kg["entities"], kg["triples"], "output/kg.json")
exporter.to_graphml(kg["graph"], "output/kg.graphml")
exporter.to_neo4j("output/kg.json", clear=False)  # needs NEO4J_URI env vars
```

---

## 6. Upload to a graph database

After building a KG, upload it to a graph database. Neo4j is currently the only
supported backend:

```python
pipe = Baseline(input_paths=["data/"], output_dir="output/")
pipe.execute()
pipe.upload_to_graph_db(backend="neo4j", clear=True)

# Or pass explicit credentials
pipe.upload_to_graph_db(
    backend="neo4j",
    clear=True,
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password",
)
```

Requires `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` environment variables
(or pass them explicitly).

To add support for another graph database (e.g. ArangoDB, Neptune), subclass
`GraphDBUploader` and register it:

```python
from kglab.kg_export.graph_db import GraphDBUploader, BACKENDS

class ArangoUploader(GraphDBUploader):
    def __init__(self, host="localhost", port=8529, **kwargs):
        self.host = host
        self.port = port

    def upload(self, json_path, clear=False):
        ...

BACKENDS["arangodb"] = ArangoUploader
```

---

## 7. Generate training data

Turn your KG into QA training pairs for LLM fine-tuning:

```python
chunks = pipe.preprocess()
kg = pipe.build_kg(chunks)
pipe.generate_training_data(kg, chunks)
# → output/training_data/kg_grounded_train.json
# → output/training_data/kg_grounded_test.json
```

---

## 8. Create a custom pipeline

Subclass `Pipeline` and override any stage. Here's a minimal example that
uses a custom entity extractor:

```python
from kglab.pipelines._base import Pipeline
from kglab._shared import Document
from kglab.preprocess import chunk, clean, dedup, load, quality
from kglab.kg_build import build_kg_into, extract, resolve
from kglab.kg_build.build import NetworkXGraphWriter
from kglab._shared.config import Ontology
from pathlib import Path

class MyPipeline(Pipeline):
    def preprocess(self) -> list[Document]:
        docs = load.from_paths(self.input_paths)
        docs = clean.normalize(docs)
        docs = quality.filter(docs, min_chars=50)
        docs = dedup.remove_duplicates(docs, method="minhash", threshold=0.85)
        chunks = chunk.by_sentence(docs, target_tokens=450)
        return chunks

    def build_kg(self, chunks: list[Document]) -> dict:
        ontology = Ontology.from_yaml(Path("configs/default_ontology.yaml"))

        # Use composed extraction with spaCy entities and ontology rules
        entities, triples = extract.with_methods(
            chunks, ontology,
            entity_method="spacy",
            relation_method="ontology_rules",
        )

        # Resolve duplicates by string matching
        resolved = resolve.by_string(entities, threshold=0.85)

        # Build the graph (storage-agnostic — swap the writer to change backend)
        writer = NetworkXGraphWriter(ontology=ontology)
        build_kg_into(writer, chunks, resolved, triples)
        graph = writer.graph

        return {"graph": graph, "entities": resolved, "triples": triples}
```

Register it so the benchmark runner can discover it:

```python
# In your own code, after defining MyPipeline:
from kglab.pipelines import PIPELINE_REGISTRY
PIPELINE_REGISTRY["my_pipeline"] = MyPipeline
```

---

## 9. Run benchmark experiments

Use `BenchmarkRunner` to run experiments and compare pipelines. You can use it
directly (no YAML required) or with a YAML config for reproducibility.

### Direct API (simplest)

```python
from kglab.benchmark_pipeline import BenchmarkRunner
from kglab.pipelines import Baseline

runner = BenchmarkRunner(
    pipeline=Baseline,
    input_paths=["data/wikipedia/"],
    output_dir="output/my_experiment/",
)
result = runner.run()

print(f"Overall score: {result.overall_score:.2f}")
print(f"Num entities:  {result.num_entities}")
print(f"Num triples:   {result.num_triples}")
```

### Compare pipelines with a benchmark test

Every benchmark test accepts multiple pipelines and scores them side by side:

```python
from kglab.benchmark_pipeline import Benchmark
from kglab.pipelines import Baseline

result = Benchmark.Dedup(dataset="benchmarks/data/dedup_gold.jsonl").run(
    pipelines={"baseline": Baseline(), "my_pipeline": MyCustomPipeline()},
)
print(result)                  # aligned per-pipeline table
print(result.best_pipeline())  # pipeline with the best score
```

### YAML config (for version-controlled experiments)

```python
from kglab.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

config = ExperimentConfig.from_yaml("experiments/kg/001_baseline/config.yaml")
runner = BenchmarkRunner.from_config(config)
result = runner.run()
```

See `experiments/kg/_template/` for an example experiment config.

---

## 10. Use pre-processing functions directly

Every stage is callable independently — you don't need a full pipeline:

```python
from kglab.preprocess import load, clean, chunk, quality, dedup

docs = load.from_paths(["data/my_articles/"])
docs = clean.normalize(docs)
docs = quality.filter(docs, min_chars=50, min_words=10)
docs = dedup.remove_duplicates(docs, method="minhash", threshold=0.85)
chunks = chunk.by_sentence(docs, target_tokens=450, overlap_tokens=60)

print(f"{len(docs)} documents → {len(chunks)} chunks")
```

Available chunkers: `chunk.by_sentence()`, `chunk.by_fixed()`, `chunk.by_semantic()`

Available dedup methods: `"minhash"`, `"exact"`, `"semantic"`

---

## Next Steps

- Read the **API Reference** (`docs/api_reference.md`) for every public class and function.
- Read the **Contributing Guide** (`CONTRIBUTING.md`) to learn how to add custom extractors, resolvers, and builders.
- Check the **example scripts** in `examples/` for more patterns.
