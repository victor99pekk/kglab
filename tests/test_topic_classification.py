import json

import torch

from ml.topic_classification import (
    ArticleExample,
    TopicClassificationConfig,
    TopicClassificationDataset,
    TopicClassificationTrainer,
    TopicTaxonomy,
    build_topic_graph,
)
from ml.topic_classification.models import get_model
from ml.topic_classification.pyg import to_hetero_data


def test_taxonomy_loads_declared_110_topics():
    taxonomy = TopicTaxonomy.from_yaml("configs/topic_taxonomy_110.yaml")

    assert len(taxonomy.topics) == 110
    assert len(taxonomy.domains) == 11
    assert taxonomy.topics["physics"].domain_id == "physical_sciences"


def test_dataset_and_graph_keep_topic_labels_out_of_message_edges(tmp_path):
    articles_path = tmp_path / "articles.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    articles_path.write_text(
        json.dumps(
            {
                "id": "article:one",
                "title": "One",
                "text": "Physics article.",
                "entities": ["entity:energy"],
                "links": ["article:two"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "article:two",
                "title": "Two",
                "text": "Astronomy article.",
                "entities": [{"id": "entity:space"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps({"article_id": "article:one", "topic_ids": ["physics"]})
        + "\n"
        + json.dumps({"article_id": "article:two", "topic_ids": ["astronomy"]})
        + "\n",
        encoding="utf-8",
    )

    dataset = TopicClassificationDataset.from_files(
        articles_path,
        labels_path,
        "configs/topic_taxonomy_110.yaml",
    )
    graph = build_topic_graph(dataset)
    indexed = graph.indexed()

    assert graph.targets == (
        ("article:one", "physics"),
        ("article:two", "astronomy"),
    )
    assert all(relation != "has_topic" for _, relation, _ in graph.edges)
    assert graph.edges[("article", "mentions", "entity")] == (
        ("article:one", "entity:energy"),
        ("article:two", "entity:space"),
    )
    assert graph.edges[("article", "links_to", "article")] == (
        ("article:one", "article:two"),
    )
    assert indexed.target_indices[0][0] == 0


def test_heterogeneous_graphsage_scores_every_article_topic_pair():
    edge_types = [
        ("article", "mentions", "entity"),
        ("entity", "mentioned_by", "article"),
        ("topic", "in_domain", "domain"),
        ("domain", "contains_topic", "topic"),
    ]
    model_cls = get_model("pyg_hetero_graphsage")
    model = model_cls(
        input_dims={"article": 8, "entity": 8, "topic": 8, "domain": 8},
        edge_types=edge_types,
        hidden_dim=16,
        num_layers=2,
    )
    features = {
        "article": torch.randn(2, 8),
        "entity": torch.randn(3, 8),
        "topic": torch.randn(4, 8),
        "domain": torch.randn(2, 8),
    }
    edge_indices = {
        ("article", "mentions", "entity"): torch.tensor([[0, 1], [0, 2]]),
        ("entity", "mentioned_by", "article"): torch.tensor([[0, 2], [0, 1]]),
        ("topic", "in_domain", "domain"): torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
        ("domain", "contains_topic", "topic"): torch.tensor(
            [[0, 0, 1, 1], [0, 1, 2, 3]]
        ),
    }

    logits = model(features, edge_indices)
    logits.sum().backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
    assert model.projections["article"].weight.grad is not None


def test_pyg_adapter_preserves_targets_without_label_edges(tmp_path):
    articles_path = tmp_path / "articles.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    articles_path.write_text(
        json.dumps(
            {
                "id": "article:one",
                "title": "One",
                "text": "Physics article.",
                "entities": ["entity:energy"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps({"article_id": "article:one", "topic_ids": ["physics"]}) + "\n",
        encoding="utf-8",
    )
    dataset = TopicClassificationDataset.from_files(
        articles_path,
        labels_path,
        "configs/topic_taxonomy_110.yaml",
    )
    graph = build_topic_graph(dataset)
    features = {
        node_type: torch.randn(len(node_ids), 8)
        for node_type, node_ids in graph.node_ids.items()
    }

    data = to_hetero_data(graph, features)

    physics_index = graph.indexed().node_index["topic"]["physics"]
    assert data["article"].y.shape == (1, 110)
    assert data["article"].y[0, physics_index] == 1
    assert all(relation != "has_topic" for _, relation, _ in data.edge_types)


def test_topic_trainer_runs_and_saves_pyg_checkpoint(tmp_path):
    taxonomy = TopicTaxonomy.from_yaml("configs/topic_taxonomy_110.yaml")
    examples = [
        ArticleExample(
            article_id=f"article:{index}",
            title=f"Article {index}",
            text="Example.",
            topic_ids=(topic_id,),
        )
        for index, topic_id in enumerate(
            ["physics", "astronomy", "chemistry", "biology", "medicine", "ecology"]
        )
    ]
    graph = build_topic_graph(TopicClassificationDataset(examples, taxonomy))
    features = {
        node_type: torch.randn(len(node_ids), 8)
        for node_type, node_ids in graph.node_ids.items()
    }
    data = to_hetero_data(graph, features)
    graph_path = tmp_path / "topic_graph.pt"
    torch.save(data, graph_path)

    trainer = TopicClassificationTrainer(
        kg_path=graph_path,
        output_dir=tmp_path / "training",
        epochs=1,
        use_gpu=False,
        model_config=TopicClassificationConfig(
            hidden_dim=8,
            num_layers=1,
            train_split=0.5,
            val_split=0.25,
        ),
    )
    result = trainer.run()
    checkpoint_path = tmp_path / "training" / "models" / "best.pt"
    restored = get_model("pyg_hetero_graphsage").load(str(checkpoint_path))
    restored_logits = restored(data.x_dict, data.edge_index_dict)

    assert checkpoint_path.exists()
    assert 0 <= result["metrics"]["micro_f1"] <= 1
    assert restored_logits.shape == (6, 110)
