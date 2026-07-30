# KG1 — spaCy Evidence Graph

KG1 is the first factual graph for Experiment 002. It uses deterministic
Wikipedia hyperlinks and spaCy named entities. Topic labels remain separate
supervision targets.

## Scope

KG1-A contains:

```text
Article -CONTAINS-> Sentence
Sentence -HAS_MENTION-> Mention
Mention -REFERS_TO-> Entity
Article -MENTIONS-> Entity
Article -LINKS_TO-> Article
Article -DESCRIBES-> Entity
Entity -HAS_SPACY_TYPE-> EntityType
```

Wikipedia hyperlinks provide the highest-confidence entity mentions. Their
target QIDs become stable entity IDs. spaCy recovers unlinked named entities and
adds source-specific type predictions. `HAS_SPACY_TYPE` does not assert that a
type is ground truth; it can be ablated independently and later compared with
Wikidata `INSTANCE_OF`. Noun chunks and entity-to-entity claims are
intentionally excluded from KG1-A.

The builder emits two graph views:

- `evidence_graph.json`: lossless sentences and mentions with revision
  provenance.
- `gnn_projection.json`: compact article, entity, and entity-type graph.

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

Then build KG1-A from that retrieved input:

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

The fixture proves the graph contract. It is not a model result. The retrieved
pilot uses each label record's pinned `article_revision_id`; current Wikipedia
text is not substituted.

## Label contract

For the first heterogeneous GraphSAGE model:

```text
article_embedding = GNN(article, KG1-A)
logits = Linear(article_embedding)  # 64 outputs
loss = multilabel_loss(logits, y_article)
```

`y_article` is loaded from `targets.jsonl` only during training/evaluation.
WikiProject templates, labels, and topic-derived category assignments cannot
enter graph nodes, node features, or message edges.
