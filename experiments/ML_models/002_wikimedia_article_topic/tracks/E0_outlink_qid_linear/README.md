# E0 — Wikimedia Outlink-QID Baseline

**Status:** planned  
**Graph:** outgoing Wikipedia links represented as a bag  
**Model:** multilabel linear classifier

Feature vector:

```text
x_article = mean(embedding(qid) for qid in article_outlinks)
```

This track reproduces Wikimedia's link-based signal as closely as available
public artifacts allow. Embedding source and dimension must be pinned before
training; no E0 embeddings exist yet.
