# E5 — Combined Heterogeneous Graph Transformer

**Status:** planned  
**Graph:** combined article-link, factual, and topic-label graph  
**Model:** HGT

Initial feature vectors:

```text
x_article = ModernBERT(article text)
x_entity  = ModernBERT(entity label + description)
x_type    = ModernBERT(type label + description)
x_topic   = ModernBERT(topic path + description)
x_domain  = ModernBERT(domain name + description)
```

HGT applies node-type and edge-type specific projections and produces
256-dimensional contextual embeddings. E5 runs only after E2–E4 ablations
identify useful relations.
