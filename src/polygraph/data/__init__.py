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
    "bench_ner": {
        "description": "CoNLL-2003 NER benchmark — TRAIN split (not a held-out evaluation)",
        "download": "polygraph.data._benchmarks:download_conll_ner",
        "enrich": None,
    },
    "bench_ner_test": {
        "description": "CoNLL-2003 NER benchmark — TEST split (held-out, literature-comparable)",
        "download": "polygraph.data._benchmarks:download_conll_ner_test",
        "enrich": None,
    },
    "bench_ner_wikiann": {
        "description": "wikiann NER benchmark — Wikipedia-derived, PER/ORG/LOC (TEST split)",
        "download": "polygraph.data._benchmarks:download_wikiann_ner",
        "enrich": None,
    },
    "bench_ner_fewnerd": {
        "description": "FewNERD NER benchmark — coarse fine-grained types (TEST split)",
        "download": "polygraph.data._benchmarks:download_fewnerd_ner",
        "enrich": None,
    },
    "bench_dedup": {
        "description": "DBLP-ACM dedup benchmark — pairwise duplicate labels",
        "download": "polygraph.data._benchmarks:download_dblp_dedup",
        "enrich": None,
    },
    "bench_resolution": {
        "description": "T2D entity resolution benchmark — table-row-to-DBpedia clusters",
        "download": "polygraph.data._benchmarks:download_t2d_resolution",
        "enrich": None,
    },
    "bench_quality": {
        "description": "TACRED relation extraction benchmark — gold relation triples",
        "download": "polygraph.data._benchmarks:download_tacred_quality",
        "enrich": None,
    },
    "bench_rag": {
        "description": "HotpotQA RAG benchmark — multi-hop QA with supporting facts",
        "download": "polygraph.data._benchmarks:download_hotpotqa_rag",
        "enrich": None,
    },
    "bench_chunking": {
        "description": "CoNLL-derived chunking benchmark — entity-span boundary integrity",
        "download": "polygraph.data._benchmarks:download_conll_chunking",
        "enrich": None,
    },
}

__all__ = ["Data", "DATASET_REGISTRY"]
