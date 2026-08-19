# Training Experiment 001 — Level-1 Topic GNN

**Date:** 2026-07-29
**Task:** multilabel article-topic link prediction
**Status:** in progress

## Research question

Does factual graph context improve English Wikipedia topic assignment over a
text-only encoder for the fixed 110-topic taxonomy?

## Boundary

- KG creation remains a `kglab` responsibility.
- Dataset construction, graph tensorization, models, and training live in
  `src/ml/topic_classification/`.
- This experiment owns configuration, label mappings, splits, results, and
  conclusions.
- Production inference code will not be added to `kglab.models` until a
  checkpoint and inference contract are validated.

## Graph contract

Message-passing node types:

- `Article`
- `Entity`
- `Topic`
- `Domain`

Message-passing relations:

- `Article -MENTIONS-> Entity`
- `Article -LINKS_TO-> Article`
- `Entity -RELATED_TO-> Entity`
- `Topic -IN_DOMAIN-> Domain`
- reverse relations for every directed relation

Supervised relation:

- `Article -HAS_TOPIC-> Topic`

`HAS_TOPIC` is stored as a target only. It must never be passed into the GNN
encoder.

## Initial model

Text embeddings are injected features. Initial graph model is a two-stage
design:

1. Any compatible English text encoder produces features for all node types.
2. PyTorch Geometric `HeteroConv` with relation-specific `SAGEConv` layers
   contextualizes those features.
3. A bilinear decoder scores every article-topic pair.

Encoder and hyperparameters are experiment choices, not core-library contracts.
Friend's PyG GCN/GAT node-classification implementation defines repository
conventions for lazy model registries and checkpoint structure. It predicts
entity types, so it is infrastructure reference rather than a task-equivalent
baseline.

## Label record

Labels remain separate from downloaded Wikipedia content:

```json
{
  "article_id": "wikipedia:en:359",
  "topic_ids": ["novels_and_fiction", "philosophy"],
  "confidence": 1.0,
  "provenance": [
    {
      "method": "manual_seed",
      "reviewer": "initial_experiment",
      "evidence": "Article describes characters and themes in Atlas Shrugged."
    }
  ]
}
```

Each line in `labels/article_topic_labels.jsonl` uses this schema.

## First milestones

1. Validate taxonomy, label records, and leakage-safe graph construction.
2. Extend Wikipedia collection with categories, hyperlinks, and Wikidata IDs.
3. Create balanced weakly-labeled corpus and reviewed validation/test subsets.
4. Generate text features.
5. Compare text-only baseline against heterogeneous GraphSAGE.
6. Record metrics and ablations in this file.

Training consumes a serialized PyG `HeteroData` artifact containing precomputed
node features. This keeps Wikipedia retrieval and BERT encoding auditable and
separate from optimizer execution.

## Results

Not run yet.
