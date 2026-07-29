# E3 — Factual Heterogeneous GraphSAGE

**Status:** planned  
**Graph:** articles, entities, entity types, and typed relations  
**Model:** heterogeneous GraphSAGE

Initial feature vectors:

```text
x_article = ModernBERT(article text)
x_entity  = ModernBERT(entity label + description)
x_type    = ModernBERT(type label + description)
```

All node types begin in a matched 768-dimensional semantic space. Typed
GraphSAGE projects them into a shared 256-dimensional graph space.
