# Experiment 002 Tracks

`002_wikimedia_article_topic` is the umbrella experiment. Every active model
track uses neural embeddings and has one stable ID and directory:

| ID | Directory | Feature vector | Status |
|---|---|---|---|
| E0 | `E0_outlink_qid_linear` | Mean pooled outlink-QID embeddings | planned |
| E1 | `E1_modernbert_text` | Hierarchical ModernBERT article embedding | planned |
| E2 | `E2_article_link_graphsage` | ModernBERT article features contextualized by GraphSAGE | planned |
| E3 | `E3_factual_hetero_graphsage` | ModernBERT article/entity/type features contextualized by heterogeneous GraphSAGE | planned |
| E4 | `E4_topic_label_gcn` | ModernBERT topic features contextualized by label GCN | planned |
| E5 | `E5_combined_hgt` | Combined typed ModernBERT features contextualized by HGT | planned |

Sparse TF-IDF is not an Experiment 002 model track. E0 and E1 are the required
embedding baselines before GNN promotion.
