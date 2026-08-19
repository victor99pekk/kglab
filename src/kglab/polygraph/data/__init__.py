"""Dataset download and enrichment — ``kglab.data``.

Provides a ``Data`` class with a clean API for downloading and enriching
supported datasets that feed into KGLab knowledge graph pipelines.

Adding a new dataset:

1. Write a module under ``kglab.data/`` with ``download_*`` and
   optionally ``enrich_*`` functions.
2. Add an entry to ``DATASET_REGISTRY`` below.

Usage::

    from kglab.data import Data, RandomSampler

    Data.list()
    Data.download("wikipedia", path="data/wikipedia/", sampler=RandomSampler(count=100))
"""

from kglab.data._api import Data
from kglab.data._wikipedia import (
    DegreeSampler,
    RandomSampler,
    SpecificSampler,
    sampler_from_config,
)

#: Supported datasets.  Each key is a short name, value is a dict with:
#:
#: * ``description`` — human-readable summary
#: * ``download`` — ``"module.path:function_name"`` for downloading
#: * ``enrich`` — ``"module.path:function_name"`` for enrichment (or ``None``)
DATASET_REGISTRY = {
    "wikipedia": {
        "description": "Wikipedia articles from HuggingFace wikimedia/wikipedia — sampler picks how to sample",
        "download": "kglab.data._wikipedia:download_wikipedia",
        "enrich": "kglab.data._wikipedia:enrich_wikipedia",
    },
    "bench_ner": {
        "description": "CoNLL-2003 NER benchmark — TRAIN split (not a held-out evaluation)",
        "download": "kglab.data._benchmarks:download_conll_ner",
        "enrich": None,
    },
    "bench_ner_test": {
        "description": "CoNLL-2003 NER benchmark — TEST split (held-out, literature-comparable)",
        "download": "kglab.data._benchmarks:download_conll_ner_test",
        "enrich": None,
    },
    "bench_ner_wikiann": {
        "description": "wikiann NER benchmark — Wikipedia-derived, PER/ORG/LOC (TEST split)",
        "download": "kglab.data._benchmarks:download_wikiann_ner",
        "enrich": None,
    },
    "bench_ner_fewnerd": {
        "description": "FewNERD NER benchmark — coarse fine-grained types (TEST split)",
        "download": "kglab.data._benchmarks:download_fewnerd_ner",
        "enrich": None,
    },
    "bench_dedup": {
        "description": "DBLP-ACM dedup benchmark — pairwise duplicate labels",
        "download": "kglab.data._benchmarks:download_dblp_dedup",
        "enrich": None,
    },
    "bench_resolution": {
        "description": "T2D entity resolution benchmark — table-row-to-DBpedia clusters",
        "download": "kglab.data._benchmarks:download_t2d_resolution",
        "enrich": None,
    },
    "bench_quality": {
        "description": "TACRED relation extraction benchmark — gold relation triples",
        "download": "kglab.data._benchmarks:download_tacred_quality",
        "enrich": None,
    },
    "bench_rag": {
        "description": "HotpotQA RAG benchmark — multi-hop QA with supporting facts",
        "download": "kglab.data._benchmarks:download_hotpotqa_rag",
        "enrich": None,
    },
    "bench_chunking": {
        "description": "CoNLL-derived chunking benchmark — entity-span boundary integrity",
        "download": "kglab.data._benchmarks:download_conll_chunking",
        "enrich": None,
    },
}

__all__ = [
    "Data",
    "DATASET_REGISTRY",
    "RandomSampler",
    "SpecificSampler",
    "DegreeSampler",
    "sampler_from_config",
]
