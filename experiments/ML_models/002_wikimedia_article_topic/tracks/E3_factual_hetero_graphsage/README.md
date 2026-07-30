# E3 — Factual Heterogeneous GraphSAGE

**Status:** KG1-A revision-matched pilot built; model planned
**Graph:** articles, linked/spaCy entities, and source-specific spaCy types
**Model:** heterogeneous GraphSAGE

Initial feature vectors:

```text
x_article = ModernBERT(article text)
x_entity  = ModernBERT(entity label + description)
x_type    = ModernBERT(type label + description)
```

All node types begin in a matched 768-dimensional semantic space. Typed
GraphSAGE projects them into a shared 256-dimensional graph space.

Graph builder and retrieval documentation:
[`../../graphs/KG1_spacy_evidence`](../../graphs/KG1_spacy_evidence/README.md).

Topic labels remain in a separate target matrix. No `HAS_TOPIC` edge enters
message passing.
