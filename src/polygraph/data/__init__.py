"""Dataset download and enrichment — ``polygraph.data``.

Provides a ``Data`` class with a clean API for downloading and enriching
supported datasets that feed into Polygraph knowledge graph pipelines.

Adding a new dataset:

1. Write a module under ``polygraph.data/`` with ``download_*`` and
   optionally ``enrich_*`` functions.
2. Add an entry to ``DATASET_REGISTRY`` below.

Usage::

    from polygraph.data import Data

    Data.list()
    Data.download("wikipedia_random", path="data/wikipedia/", count=100, enrich=True)
"""

from polygraph.data._api import Data

#: Supported datasets.  Each key is a short name, value is a dict with:
#:
#: * ``description`` — human-readable summary
#: * ``download`` — ``"module.path:function_name"`` for downloading
#: * ``enrich`` — ``"module.path:function_name"`` for enrichment (or ``None``)
DATASET_REGISTRY = {
    "wikipedia_random": {
        "description": "Random Wikipedia articles from HuggingFace wikimedia/wikipedia",
        "download": "polygraph.data._wikipedia:download_wikipedia",
        "enrich": "polygraph.data._wikipedia:enrich_wikipedia",
    },
    # "wikimedia_topic_labels": {
    #     "description": "Wikimedia article-topic labels from Figshare (Experiment 002)",
    #     "download": "polygraph.data._wikimedia_topics:download_topic_labels",
    #     "enrich": None,
    # },
}

__all__ = ["Data", "DATASET_REGISTRY"]
