# API Reference

Complete reference for all public classes, functions, and types in Polygraph.

---

## Core Types (`polygraph._shared`)

### `Document`

```python
from polygraph._shared import Document
```

The universal data currency between all pipeline stages. Represents a raw
source file, a cleaned text, or a chunk.

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Full text body |
| `source` | `str` | Origin path, URL, or identifier |
| `doc_id` | `str` | Stable unique ID (used for dedup, provenance) |
| `language` | `str` | ISO 639-1 code (default `"en"`) |
| `metadata` | `dict` | Arbitrary stage-specific annotations |

---

### `Ontology`

```python
from polygraph._shared.config import Ontology
```

Defines the KG schema: which entity types, relations, and attributes exist.

| Method | Returns | Description |
|---|---|---|
| `Ontology.from_yaml(path)` | `Ontology` | Load from a YAML file |
| `.get_entity_type_names()` | `list[str]` | Entity type labels |
| `.get_relation_patterns()` | `list[tuple]` | `(domain, range, predicate, symmetric)` tuples for rule-based extraction |
| `.get_structural_relation_types()` | `dict` | Structural predicates → `(domain, range)` |

See `configs/default_ontology.yaml` for the default schema.

---

## Pipelines (`polygraph.pipelines`)

### `Pipeline` (abstract base)

```python
from polygraph.pipelines._base import Pipeline
```

Base class for all KG generation pipelines. Subclass and override stage methods
to create variants.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(input_paths, output_dir, **kwargs)` | Set up input/output paths |
| `preprocess()` | `() -> list[Document]` | **Abstract.** Raw files → clean chunks. |
| `build_kg(chunks)` | `(list[Document]) -> dict` | **Abstract.** Chunks → KG dict with `graph`, `entities`, `triples`. |
| `evaluate(kg, llm_client=None)` | `(dict, callable?) -> dict` | Run quality metrics. Pass `llm_client` for accuracy eval. |
| `export(kg)` | `(dict) -> None` | Export to JSON (and GraphML if `graphml=True` in config). |
| `upload_to_neo4j(clear=False)` | `(bool) -> None` | Upload exported KG JSON to Neo4j. |
| `generate_training_data(kg, chunks)` | `(dict, list[Document]) -> None` | Generate QA pairs for LLM fine-tuning. |
| `execute()` | `() -> None` | Full pipeline: preprocess → build → evaluate → export. |

---

### `Baseline`

```python
from polygraph.pipelines import Baseline
```

Standard pipeline with configurable extraction, resolution, and build methods.
Accepts typed config objects in addition to raw `**kwargs`.

```python
Baseline(
    input_paths=["data/"],
    output_dir="output/",
    extraction=ExtractionConfig(mode="composed", entity_method="spacy"),
    resolution=ResolutionConfig(method="string", threshold=0.85),
    build=BuildConfig(method="networkx", graphml=False),
)
```

---

### `PIPELINE_REGISTRY`

```python
from polygraph.pipelines import PIPELINE_REGISTRY
```

Registry mapping variant name strings → `Pipeline` subclasses. Add your custom
pipeline here for CLI and `BenchmarkRunner` discovery:

```python
PIPELINE_REGISTRY["my_variant"] = MyPipeline
```

---

## Stage Configuration (`polygraph._shared.stage_config`)

### `ExtractionConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"composed"` | `"composed"` or `"joint"` |
| `entity_method` | `str` | `"spacy"` | Entity extractor name |
| `relation_method` | `str` | `"ontology_rules"` | Relation extractor name |
| `joint_method` | `str` | `"graphgen"` | Joint extractor name |
| `entity_options` | `dict \| None` | `None` | Options passed to entity method |
| `relation_options` | `dict` | `{}` | Options passed to relation method |
| `options` | `dict` | `{}` | Options for joint methods |
| `document_relation` | `DocumentRelationConfig` | — | Doc-to-doc relation settings |

### `ResolutionConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"string"` | `"string"` or `"embedding"` |
| `threshold` | `float` | `0.85` | Similarity threshold for merging |
| `options` | `dict` | `{}` | Additional method options |

### `BuildConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"networkx"` | Graph construction backend |
| `graphml` | `bool` | `False` | Also export GraphML |

---

## Pre-processing (`polygraph.preprocess`)

```python
from polygraph.preprocess import load, clean, chunk, quality, dedup
```

Each stage is a callable namespace. Available methods per stage:

### `load`

| Function | Description |
|---|---|
| `load.from_paths(paths)` | Load all `.jsonl`/`.txt` files from paths → `list[Document]` |

### `clean`

| Function | Description |
|---|---|
| `clean.normalize(docs)` | Normalize whitespace, encoding, control chars |

### `quality`

| Function | Description |
|---|---|
| `quality.filter(docs, min_chars, min_words)` | Remove low-quality documents |

### `dedup`

| Function | Description |
|---|---|
| `dedup.remove_duplicates(docs, method, threshold)` | Remove near-duplicate documents. Methods: `"minhash"`, `"exact"`, `"semantic"`. |

### `chunk`

| Function | Description |
|---|---|
| `chunk.by_sentence(docs, target_tokens, overlap_tokens)` | Split by sentence boundaries |
| `chunk.by_fixed(docs, chunk_size, overlap)` | Split by fixed token count |
| `chunk.by_semantic(docs, target_tokens, overlap_tokens)` | Split at semantic boundaries |

---

## KG Build (`polygraph.kg_build`)

### `extract.with_methods()`

```python
from polygraph.kg_build import extract

entities, triples = extract.with_methods(
    chunks, ontology,
    entity_method="spacy",
    relation_method="ontology_rules",
    entity_options={"model_name": "en_core_web_sm"},
    relation_options={},
)
# Returns: (list[dict], list[tuple])
```

### `extract.jointly()`

```python
entities, triples = extract.jointly(
    chunks, ontology,
    method="graphgen",
    options={"model": "gpt-4o"},
)
```

### `resolve.by_string()` / `resolve.by_embedding()`

```python
from polygraph.kg_build import resolve

# Fast token-overlap resolution
resolved = resolve.by_string(entities, threshold=0.85)

# Semantic embedding-based resolution (requires sentence-transformers)
resolved = resolve.by_embedding(
    entities, threshold=0.85,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
```

### `build.from_resolved()`

```python
from polygraph.kg_build import build

graph = build.from_resolved(resolved, triples, method="networkx", ontology=ontology)
# Returns: networkx.DiGraph
```

---

## KG Evaluation (`polygraph.kg_eval`)

```python
from polygraph.kg_eval import metrics, structural
from polygraph.kg_eval.metrics import AccuracyEvaluator

# Basic quality metrics (no LLM needed)
report = metrics.evaluate(graph, entities, triples)

# Structural audit (ontology compliance, connectivity, etc.)
audit = structural.run(graph, entities, triples, ontology_path="configs/default_ontology.yaml")

# Accuracy evaluation (requires LLM callable)
eval = AccuracyEvaluator(llm_client=my_llm_function)
accuracy = eval.evaluate(graph, entities, triples)
```

---

## KG Export (`polygraph.kg_export`)

```python
from polygraph.kg_export import GraphExporter, exporter

# Class API — export to multiple formats at once
ge = GraphExporter()
ge.export(graph, entities, triples, output_dir, formats=["json", "graphml", "neo4j_csv", "rdf"])

# Function API — one call per format
exporter.to_json(graph, entities, triples, "output/kg.json")
exporter.to_graphml(graph, "output/kg.graphml")

# Graph database upload (Neo4j is the only currently supported backend)
exporter.to_graph_db("output/kg.json", backend="neo4j", clear=False)
exporter.to_graph_db("output/kg.json", backend="neo4j",
                     uri="bolt://localhost:7687", user="neo4j", password="secret")
```

| Format | Key | Description |
|---|---|---|
| `json` | `"json"` | Full KG JSON (default) |
| `graphml` | `"graphml"` | GraphML for Gephi/Cytoscape |
| `neo4j_csv` | `"neo4j_csv"` | CSV files for Neo4j bulk import |
| `rdf` | `"rdf"` | RDF/Turtle triples |
| `cytoscape` | `"cytoscape"` | Cytoscape.js JSON |

### Graph Database Upload (`polygraph.kg_export.graph_db`)

```python
from polygraph.kg_export.graph_db import Neo4jUploader, GraphDBUploader, BACKENDS

# Direct uploader usage
uploader = Neo4jUploader(uri="bolt://localhost:7687", user="neo4j", password="secret")
uploader.upload("output/kg.json", clear=True)

# Add a new backend
class ArangoUploader(GraphDBUploader):
    def upload(self, json_path, clear=False):
        ...

BACKENDS["arangodb"] = ArangoUploader
```

---

## Benchmark Runner (`polygraph.benchmark_pipeline`)

### Direct API (recommended)

```python
from polygraph.benchmark_pipeline import BenchmarkRunner
from polygraph.pipelines import Baseline

# Single pipeline run
runner = BenchmarkRunner(
    pipeline=Baseline,
    input_paths=["data/wikipedia/"],
    output_dir="output/my_exp/",
)
result = runner.run()

# Compare two pipelines side by side
results = BenchmarkRunner.compare(
    baseline=Baseline,
    variant=MyPipeline,
    input_paths=["data/wikipedia/"],
    output_dir="output/comparison/",
)
```

### `BenchmarkRunner`

| Method | Description |
|---|---|
| `BenchmarkRunner(pipeline, input_paths, output_dir, ...)` | Direct constructor — no YAML needed |
| `BenchmarkRunner.from_config(config)` | Create from an `ExperimentConfig` (YAML) |
| `BenchmarkRunner.compare(baseline, variant, input_paths, output_dir)` | Run two pipelines side by side |
| `.run()` → `BenchmarkResult` | Execute and return results |

### `BenchmarkResult`

| Property | Type | Description |
|---|---|---|
| `.overall_score` | `float` | Overall KG quality (0–1) |
| `.num_entities` | `int` | Number of entities |
| `.num_triples` | `int` | Number of triples |
| `.metrics` | `dict` | Raw metrics dict |
| `.structural_audit` | `dict` | Ontology compliance audit |
| `.artifacts` | `dict` | Paths to generated files |
| `.wall_time_s` | `float` | Elapsed time (via `metrics`) |

### YAML config (for reproducibility)

```python
from polygraph.benchmark_pipeline import BenchmarkRunner, ExperimentConfig

config = ExperimentConfig.from_yaml("experiments/kg/001_baseline/config.yaml")
runner = BenchmarkRunner.from_config(config)
result = runner.run()
```

See `experiments/kg/_template/config.yaml` for the full config schema.

---

## Data Download (`polygraph.data`)

```python
from polygraph.data import Data

Data.download("wikipedia_random", path="data/wikipedia/articles.jsonl", enrich=True)
```

---

## Entity & Relation Types (`polygraph.kg_build.extract`)

### `Entity`

```python
from polygraph.kg_build.extract import Entity
```

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Entity surface form |
| `label` | `str` | Entity type (e.g. `"PERSON"`) |
| `mentions` | `list[str]` | All text mentions |
| `confidence` | `float` | Extraction confidence (0–1) |
| `description` | `str` | Optional description |
| `source` | `str` | Source chunk ID |
| `embedding` | `list[float] \| None` | Vector embedding (if computed) |

### `EntityExtractor` (ABC)

Abstract base for entity extraction methods. Implement `extract(text) -> list[Entity]`.

### `RelationExtractorMethod` (ABC)

Abstract base for relation extraction methods. Implement `extract(text, entities, source_chunk_id) -> list[tuple]`.

### `JointExtractor` (ABC)

Abstract base for joint entity+relation extraction. Implement `extract(text, source_chunk_id) -> tuple[list[Entity], list[tuple]]`.

---

## Base Evaluator (`polygraph.kg_eval`)

### `BaseEvaluator` (ABC)

Abstract base for custom evaluators. Implement `evaluate(graph, entities, triples) -> dict`.

```python
from polygraph.kg_eval._base import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def evaluate(self, graph, entities, triples):
        return {"my_metric": compute_something(graph, entities, triples)}
```
