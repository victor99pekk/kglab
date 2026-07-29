# E4 — Topic-Label GCN

**Status:** planned  
**Graph:** topic hierarchy plus train-only topic co-occurrence  
**Model:** label GCN

Feature vector:

```text
x_topic_initial = ModernBERT(topic path + description)
x_topic_graph = GCN(x_topic_initial, hierarchy + train co-occurrence)
logit(article, topic) = score(x_article, x_topic_graph)
```

Only training labels may construct co-occurrence edges. Validation and test
labels remain hidden.
