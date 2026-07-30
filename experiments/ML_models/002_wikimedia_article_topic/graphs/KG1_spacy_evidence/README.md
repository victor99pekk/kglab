# KG1-A.2 — spaCy Feature Evidence Graph

KG1-A.2 is the first factual graph for Experiment 002. It uses deterministic
Wikipedia hyperlinks, sentence-sized chunks, and spaCy named entities. Topic
labels remain separate supervision targets.

## Scope

KG1-A.2 contains:

```text
Article -HAS_CHUNK-> Chunk
Chunk -NEXT-> Chunk
Chunk -HAS_MENTION-> Mention
Mention -REFERS_TO-> Entity
Chunk -MENTIONS-> Entity
Article -MENTIONS-> Entity
Article -LINKS_TO-> Article
Article -DESCRIBES-> Entity
```

Wikipedia hyperlinks provide the highest-confidence entity mentions. Their
target QIDs become stable entity IDs. spaCy recovers unlinked named entities and
adds source-specific type predictions to mentions. The graph aggregates those
predictions into `spacy_types`, `spacy_type_counts`, and
`primary_spacy_type` entity attributes.

Raw spaCy labels are not nodes. This avoids high-degree type hubs and preserves
the fact that NER labels are extraction evidence rather than canonical entity
facts. Later canonical types may use curated `EntityClass` nodes backed by
Wikidata `INSTANCE_OF` and `SUBCLASS_OF`.

Chunks currently use one sentence per chunk so mention offsets and provenance
remain deterministic. The later training pipeline may replace this with
token-window chunks without changing the graph contract. Noun chunks and
entity-to-entity claims remain intentionally excluded from KG1-A.2.

The builder emits two graph views:

- `evidence_graph.json`: lossless chunks and mentions with revision
  provenance.
- `gnn_projection.json`: article, chunk, and entity graph; mention nodes are
  removed while their spaCy type evidence remains on entities.

Labels are written to `targets.jsonl`. There is no `HAS_TOPIC` edge.

## Pilot

Run experiment-local tests:

```bash
uv run --frozen pytest \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/test_kg1.py
```

Build the bundled deterministic fixture:

```bash
uv run --frozen \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/kg1_pipeline.py \
  build
```

Retrieve a small revision-matched pilot from the prepared Wikimedia labels:

```bash
uv run --frozen \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/retrieve_pilot.py \
  --per-topic 1 \
  --max-articles 8
```

Then build KG1-A.2 from that retrieved input:

```bash
uv run --frozen \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/kg1_pipeline.py \
  build \
  --input experiments/ML_models/002_wikimedia_article_topic/artifacts/kg1_revision_pilot/retrieved_articles.jsonl \
  --output experiments/ML_models/002_wikimedia_article_topic/artifacts/kg1_revision_pilot
```

Generated pilot artifacts:

```text
experiments/ML_models/002_wikimedia_article_topic/artifacts/kg1_spacy_pilot/
  evidence_graph.json
  gnn_projection.json
  label_graph.json
  targets.jsonl
  summary.json
  visualization.svg
```

## Neo4j

Adapt the compact GNN projection to the repository uploader contract:

```bash
uv run --frozen \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/export_neo4j.py
```

Upload through the existing repository uploader without clearing other data:

```bash
uv run --frozen python -c \
  "from dotenv import load_dotenv; load_dotenv(); from polygraph.kg_export.neo4j.upload import upload_from_output; upload_from_output('experiments/ML_models/002_wikimedia_article_topic/artifacts/kg1_revision_pilot', clear=False)"
```

Do not use root `make neo4j-upload` for this pilot: that target currently passes
`clear=True`.

If Neo4j does not have APOC installed, use the experiment-local native Cypher
uploader:

```bash
uv run --frozen \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/upload_neo4j.py \
  replace

uv run --frozen \
  experiments/ML_models/002_wikimedia_article_topic/graphs/KG1_spacy_evidence/upload_neo4j.py \
  inspect
```

`replace` atomically deletes only nodes tagged `KG1-A` or `KG1-A.2`, then
uploads the new KG1-A.2 payload. Unrelated Neo4j graph data is preserved. The
transaction rolls back both deletion and insertion if upload fails.

Configured Neo4j lacks APOC, so repository uploader currently writes nodes but
cannot write relationships. Native uploader completes same contract using fixed,
whitelisted Cypher relationship types. `inspect` writes
`artifacts/kg1_revision_pilot/neo4j_confirmation.json`.

The fixture proves the graph contract. It is not a model result. The retrieved
pilot uses each label record's pinned `article_revision_id`; current Wikipedia
text is not substituted.

The confirmed KG1-A.2 pilot contains 277 nodes and 533 unique relationships
(567 weighted source edges). Neo4j inspection reports zero remaining KG1-A
nodes or relationships. Generated JSON, SVG, PNG, targets, and confirmation
files are saved under `artifacts/kg1_revision_pilot/`, which is intentionally
Git-ignored.

## Model contract

KG1-A.2 prepares inputs for the first heterogeneous GraphSAGE model:

```text
x_chunk = ModernBERT(chunk text)
x_article = attention_pool(x_chunk)
x_entity = concat(
    ModernBERT(entity label + description),
    multi_hot(log1p(spacy_type_counts)),
)

article_embedding = GNN(article, KG1-A.2)
logits = Linear(article_embedding)  # 64 outputs
loss = multilabel_loss(logits, y_article)
```

`y_article` is loaded from `targets.jsonl` only during training/evaluation.
WikiProject templates, labels, and topic-derived category assignments cannot
enter graph nodes, node features, or message edges.

The graph builder does not train a GNN. There is currently no GraphSAGE
checkpoint or metric result; ModernBERT feature generation and the training
runner are future work.
