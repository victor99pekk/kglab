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
| `preprocess()` | `() -> list[Document] \| PreprocessResult` | **Abstract.** Raw files → clean chunks (may carry pre-extracted entities/triples). |
| `build_kg(chunks)` | `(list[Document] \| PreprocessResult) -> dict` | **Abstract.** Chunks → KG dict with `graph`, `entities`, `triples`. |
| `export(kg, config=None)` | `(dict, ExportConfig?) -> None` | Export per `ExportConfig.formats` (JSON default; GraphML/Neo4j opt-in). |
| `upload_to_graph_db(backend="neo4j", clear=False)` | `(str, bool) -> None` | Upload exported KG JSON to a graph DB (Neo4j only). |
| `upload_to_neo4j(clear=False)` | `(bool) -> None` | Alias for `upload_to_graph_db(backend="neo4j")`. |
| `generate_training_data(kg, chunks)` | `(dict, list[Document]) -> None` | Generate QA pairs for LLM fine-tuning. |
| `execute(input_paths=None, output_dir=None, export_config=None, cache=False, force=False)` | `(...) -> dict` | Full pipeline: preprocess → build → export. Returns the built KG. |

> **Evaluation is not part of `Pipeline`.** Pipelines only build + export.
> Score the returned KG with `polygraph.kg_eval.evaluate_kg(kg, output_dir=...)`,
> or use `BenchmarkRunner`, which evaluates automatically.

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

### `Semantic` / `StreamingPipeline`

Additional bundled variants:

- `Semantic(input_paths=..., output_dir=...)` — Baseline with semantic chunking/resolution defaults.
- `StreamingPipeline(...)` — memory-bounded, disk-backed pipeline for large corpora.

`PIPELINE_REGISTRY` already maps `"baseline"`, `"surface"` (legacy alias), and `"semantic"`.

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

### `PreprocessConfig` / `PreprocessStage`

Tiered preprocessing control:

- **Tier 1 knobs**: `clean_enabled`, `link_normalize_enabled`, `quality_min_chars` (200), `quality_min_words` (40), `doc_dedup_method` (`"layered"`), `doc_dedup_threshold` (0.85), `chunk_method` (`"semantic"`), `chunk_target_tokens` (450), `chunk_overlap_tokens` (60), `chunk_dedup_method`, `chunk_dedup_threshold`.
- **Tier 2** — explicit `stages=[PreprocessStage("load"), PreprocessStage("chunk", "semantic", options={...}), ...]` (overrides knobs).

`PreprocessStage(name, method="default", enabled=True, options={})` — one named pipeline step.

### `EvalConfig` / `ExportConfig` / `LinkingConfig`

| Config | Fields | Default |
|---|---|---|
| `EvalConfig` | `quality_enabled`, `structural_enabled`, `accuracy_enabled` | `True, True, False` |
| `ExportConfig` | `formats` (`"json"`, `"graphml"`, `"neo4j"`), `neo4j_clear` | `["json"], False` |
| `LinkingConfig` | `method`, `enabled`, `options` | `"", False, {}` |

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
| `dedup.remove_duplicates(docs, method="layered", threshold=0.85)` | Remove near-duplicate documents. Methods: `"layered"` (default), `"minhash"`, `"semantic"`. |

### `chunk`

| Function | Description |
|---|---|
| `chunk.by_sentence(docs, target_tokens=450, overlap_tokens=60)` | Split by sentence boundaries |
| `chunk.by_fixed(docs, size=500, overlap=100)` | Split by fixed token count |
| `chunk.by_semantic(docs, target_tokens=450, overlap_tokens=60)` | Split at semantic boundaries |

### `link`

| Function | Description |
|---|---|
| `link.normalize_links(docs)` | Normalize hyperlink markup in documents |

Also available: `load.stream(paths)` (streaming loader) and the class API — `DataLoader`, `TextCleaner`, `QualityFilter`, `Deduplicator`, `SentenceChunker`, `SemanticChunker`, `TextChunker`, `Preprocessor`, `DefaultPreprocessor`.

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

### `link.entities()`

```python
from polygraph.kg_build import link

linked = link.entities(resolved, method="wikidata")  # adds kb_id / kb_source
```

Registry: `LINKER_REGISTRY`; base contract: `EntityLinker` (`linker.link(entity)`). Add backends via `LINKER_REGISTRY["my_kb"] = MyLinker`.

### `build.from_resolved()`

```python
from polygraph.kg_build import build

# In-memory (default)
graph = build.from_resolved(resolved, triples, method="networkx", ontology=ontology)
# Returns: networkx.DiGraph

# File-backed (for large graphs — avoids OOM)
graph = build.from_resolved(resolved, triples, method="sqlite")
# Returns: SQLiteGraph (disk-resident, same API as nx.DiGraph)
```

| Method | Backend | RAM usage | File |
|---|---|---|---|
| `"networkx"` | NetworkX | All nodes + edges | None |
| `"sqlite"` | SQLite | Queried data only | `knowledge_graph.db` |

To use SQLite with a pipeline:

```python
from polygraph._shared.stage_config import BuildConfig

pipe = Baseline(..., build=BuildConfig(method="sqlite"))
```

---

## KG Evaluation (`polygraph.kg_eval`)

```python
from polygraph.kg_eval import metrics, structural, evaluate_kg
from polygraph.kg_eval.metrics import AccuracyEvaluator

# One-shot: quality + structural, writes metrics.json
# (this is what BenchmarkRunner uses after pipeline.execute())
report = evaluate_kg(kg, output_dir="output/")

# Or compose the pieces yourself:
# Basic quality metrics (no LLM needed)
report = metrics.evaluate(graph, entities, triples)
# Attribute coverage score (no LLM)
metrics.completeness(entities)

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
exporter.to_neo4j("output/kg.json", clear=False)  # alias for to_graph_db(backend="neo4j")

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
```

### `BenchmarkRunner`

| Method | Description |
|---|---|
| `BenchmarkRunner(pipeline, input_paths, output_dir, ...)` | Direct constructor — no YAML needed |
| `BenchmarkRunner.from_config(config)` | Create from an `ExperimentConfig` (YAML) |
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

### `Benchmark` (stage benchmarks)

```python
from polygraph.benchmark_pipeline import Benchmark

result = Benchmark.Dedup(dataset="benchmarks/data/dedup_gold.jsonl").run(
    pipelines={"baseline": Baseline(), "semantic": Semantic()}
)
```

Stages: `Dedup`, `Resolution`, `Chunking`, `Extraction`, `Quality`, `RAG`. Pass **any number of pipelines** to `pipelines={...}` to compare them head-to-head (this is the replacement for the removed `BenchmarkRunner.compare`). Runners return `StageResult` (`.best_pipeline()`, `.to_dict()`); helpers: `write_report()`, `format_stage(s)`.

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
from polygraph.data import Data, DegreeSampler, RandomSampler, SpecificSampler

Data.list()                                                            # available datasets
Data.download("wikipedia", sampler=RandomSampler(count=20))
Data.download("wikipedia", path="data/wikipedia/articles.jsonl", force=False)
Data.download("wikipedia", sampler=SpecificSampler(urls=[...]))
Data.download("wikipedia", sampler=DegreeSampler(count=50, target_degree=5.0))
Data.enrich("wikipedia", input_path="data/wikipedia/articles.jsonl")
```

`Data.download("wikipedia", ...)` samples articles via a sampler object:
`RandomSampler` (reservoir-samples from HuggingFace, default), `SpecificSampler`
(explicit URLs or a URL file), or `DegreeSampler` (grows a connected, link-rich
set to a target average hyperlink degree). Shared options — `language`, `snapshot`,
`append` — stay at the top level. Downloads are automatically enriched with
outgoing Wikipedia hyperlinks; `Data.enrich(...)` re-enriches an existing file.

`Data.info(name)` returns dataset metadata. `DATASET_REGISTRY` holds all entries: Wikipedia + NER/dedup/resolution/quality/chunking/RAG benchmarks.

---

## Fine-tuning & Models

```python
from polygraph.finetune.dataset import QADatasetGenerator

gen = QADatasetGenerator(language="en", seed=42, max_hops=3, test_split=0.2)
pairs = gen.generate_from_kg(graph, entities, triples)  # {split: [QA pairs]}
```

`polygraph.models` provides pluggable model backends (node classification, entity resolution) via `polygraph.models.registry`.

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
| `attributes` | `dict` | Arbitrary entity attributes |
| `confidence` | `float` | Extraction confidence (0–1) |
| `description` | `str` | Optional description |
| `source` | `str` | Source chunk ID |
| `embedding` | `list[float] \| None` | Vector embedding (if computed) |
| `node_id` | `str` | Optional explicit node ID |
| `source_chunk_ids` | `list[str]` | Chunks that mention this entity |

Properties: `.id` (resolved node ID), `.aliases` (normalized mentions), `.displayName`.

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
