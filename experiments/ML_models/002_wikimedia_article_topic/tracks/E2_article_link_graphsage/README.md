# E2 — Article-Link GraphSAGE

**Status:** planned  
**Graph:** `Article -LINKS_TO-> Article`  
**Model:** inductive GraphSAGE

Initial feature vector:

```text
x_article_initial = hierarchical ModernBERT article embedding
x_article_graph = GraphSAGE(x_article_initial, article_links)
```

The matched E1 article embedding must remain identical before message passing,
so E2 measures graph value rather than encoder changes.
