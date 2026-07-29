# Training Experiment 002 — Wikimedia Article Topics

**Date:** 2026-07-29  
**Task:** multilabel article-topic classification  
**Status:** data prepared; embedding baselines and KG1 not implemented  
**Labels:** Wikimedia original 64-topic taxonomy

## Research question

Can BERT text features plus an inductive Wikipedia/Wikidata graph improve
article-topic classification over Wikimedia's link-only approach and a matched
text-only encoder?

This experiment intentionally reproduces Wikimedia's public classification
problem before attempting the repository's custom 110-topic taxonomy.

## Classification target

One Wikipedia article is one supervised example. The output is a 64-dimensional
multilabel vector.

- Articles are classified.
- Chunks are evidence used to construct article text features.
- Entities, entity types, topics, and domains are context nodes.
- WikiProject assessments generate labels and must not enter the graph as
  features.

## Public label source

Wikimedia researchers published the CC0 dataset
[Wikipedia Articles and Associated WikiProject Templates][figshare]. The
English archive contains article revision IDs, Wikidata QIDs, WikiProject
templates, sitelinks, and derived topic labels.

Pinned source:

- Figshare article `10248344`, version 4
- English archive file `22969217`
- Published 2020-06-08
- English archive size: `316930407` bytes
- English archive MD5: `3dd89dd7d5ab645731841c996aec4e0b`

The labels are community-derived traces rather than exhaustive manual
annotations. They are suitable for weak training labels. Final validation and
test conclusions still require an audited subset.

## Experiment tracks

Canonical per-track files live under [`tracks/`](tracks/README.md). E0–E5 all
use neural embeddings. Non-embedding classifiers are outside experiment scope.

### E0 — Wikimedia-style outlink baseline

Represent an article using its outgoing Wikipedia links mapped to Wikidata
QIDs. Pool linked-item embeddings and train a multilabel linear classifier.

Purpose: closest feasible reproduction of Wikimedia's published link-based
classifier.

### E1 — ModernBERT embedding baseline

Encode title, lead, and body chunks with a BERT-family encoder. Attention-pool
chunks into one article representation and predict 64 topics.

Purpose: establish matched learned-embedding baseline.

### E2 — article-link GraphSAGE

Use article text embeddings as node features and Wikipedia hyperlinks as graph
edges. Train GraphSAGE for inductive article classification.

Purpose: measure graph value for genuinely new article nodes.

### E3 — factual heterogeneous GraphSAGE

Add article-to-entity, entity-to-type, and type hierarchy edges sourced from
Wikipedia links and Wikidata.

Purpose: test whether factual context improves article embeddings.

### E4 — topic-label GCN

Encode the 64 Wikimedia topics as label nodes. Add taxonomy hierarchy and
train-only topic co-occurrence edges.

Purpose: model label dependencies without requiring an article to be present in
the training graph.

### E5 — combined HGT

Combine article, entity, type, topic, and domain nodes with typed relations.

Purpose: final high-capacity candidate after simpler ablations prove useful.

## GNN contract

Initial node features:

```text
x_article: [article_count, text_dim]
x_entity:  [entity_count, text_dim]
x_type:    [type_count, text_dim]
x_topic:   [64, text_dim]
x_domain:  [4, text_dim]
```

Candidate message edges:

```text
Article -LINKS_TO-> Article
Article -MENTIONS-> Entity
Entity  -INSTANCE_OF-> Type
Type    -SUBCLASS_OF-> Type
Topic   -IN_DOMAIN-> Domain
Topic   -RELATED_TO-> Topic
```

Supervision:

```text
y_article: [article_count, 64]
```

`Article -HAS_TOPIC-> Topic` is a target only. It is never a message-passing
edge.

Output:

```text
article_logits: [article_count, 64]
article_probabilities = sigmoid(article_logits)
```

## Existing-node and new-node evaluations

Two results must remain separate:

1. **Inductive, primary:** test articles are absent during training. At
   inference, a new article receives text features and an available one-hop
   neighborhood.
2. **Transductive, secondary:** test articles are present as unlabeled graph
   nodes during training. This measures embedding refinement inside a fixed
   graph.

## Data preparation

Validate pinned metadata and taxonomy:

```bash
make wikimedia-topic-labels-validate
```

Download the 316.9 MB English label archive:

```bash
make wikimedia-topic-labels-download
```

Normalize records and create deterministic QID-grouped splits:

```bash
make wikimedia-topic-labels-prepare
```

Generated files stay under this experiment's ignored `artifacts/` directory.
Reusable implementation lives at
`tools/data_retrieval/prepare_wikimedia_topic_labels.py`.

## Data gates

1. ~~Validate all source labels against the frozen 64-topic taxonomy.~~
2. ~~Profile label counts and split sizes.~~
3. Profile label cardinality and co-occurrence.
4. Audit at least 25 validation and 50 test positives per topic.
5. Retrieve revision-matched text and outlinks for a balanced pilot.
6. Run E0 and E1 before adding GNNs.
7. Promote graph models only when matched inductive evaluation is available.

## Run results

Full public-label preparation completed on 2026-07-29.

| Item | Value |
|---|---:|
| Normalized labeled articles | 5,485,430 |
| Train articles | 4,936,589 |
| Validation articles | 109,308 |
| Test articles | 439,533 |
| Skipped source records | 310,925 |
| Taxonomy labels observed | 64 / 64 |

Artifacts:

- `artifacts/raw/labeled_enwiki_with_topics_metadata.json.bz2`
- `artifacts/prepared/article_topic_labels.jsonl`
- `artifacts/prepared/summary.json`

E0–E5 model metrics are not available yet. Those tracks currently specify
research designs; no training runner implements them.

## Next KG1 — traditional NLP

KG1 will begin with deterministic spaCy and rule-based extraction:

```text
Article -MENTIONS-> Entity
Entity  -HAS_TYPE-> spaCy entity type
Entity  -RELATED_TO-> Entity
Topic   -IN_DOMAIN-> Domain
```

`RELATED_TO` must have sentence evidence and a named dependency/rule pattern.
WikiProject/category label sources cannot become KG features.
`Article -HAS_TOPIC-> Topic` remains supervision only.

## Approval boundary

Label preparation is now a shared retrieval tool. Extending `src/ml`, changing
dependencies, or writing other shared data still requires explicit approval.

[figshare]: https://figshare.com/articles/dataset/Wikipedia_Articles_and_Associated_WikiProject_Templates/10248344
