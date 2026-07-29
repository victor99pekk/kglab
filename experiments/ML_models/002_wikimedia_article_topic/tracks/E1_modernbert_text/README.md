# E1 — ModernBERT Embedding Baseline

**Status:** planned  
**Graph:** none  
**Model:** hierarchical multilabel text classifier

Feature vector:

```text
h_chunk = ModernBERT(title + lead/body chunk)
x_article = attention_pool(h_chunk_1, ..., h_chunk_n)
```

Planned encoder: `answerdotai/ModernBERT-base`. Article representation is dense
and fine-tuned for 64-label prediction. Revision-matched article text is not
retrieved yet.
