# KGLab Paper — Motivation & Research Direction

> Modeled after *"Fast Graph Representation Learning with PyTorch Geometric"* (Fey & Lenssen, ICLR 2019)

---

## The Gap

**Knowledge graph construction from text has no standard library.**

- GNNs have PyTorch Geometric, DGL, Spektral — standardized, modular, benchmarked.
- NLP has HuggingFace Transformers, spaCy, Stanza — standardized, modular, benchmarked.
- KG construction has... ad-hoc scripts glued together per project. No shared abstractions. No fair comparisons. No standard evaluation.

Every research group that builds KGs from text reinvents: chunking, entity extraction, relation extraction, entity resolution, graph building, evaluation, and export. Results are incomparable because every pipeline is different.

## The Opportunity

KGLab fills this gap. It provides:

1. **A modular, end-to-end pipeline** — preprocessing → extraction → resolution → building → evaluation → export → QA generation — with swappable components at every stage via a registry pattern.

2. **A standardized KG quality evaluation framework** — multidimensional metrics (completeness, consistency, duplication, structural health, semantic accuracy via LLM judge, triple classification F1) that define what "good" means for a KG.

3. **Reproducible benchmarking** — `ExperimentConfig` + `BenchmarkRunner` enable apples-to-apples comparisons across extraction methods, ontologies, and resolution strategies, controlled by a single YAML file.

4. **A plugin architecture** — new extractors, resolvers, or exporters can be added without touching existing code, lowering the barrier for community contributions.

## Why Now

- **LLMs need better training data.** KG-structured data demonstrably reduces hallucination (0% vs 94% in initial experiments).
- **LLM-based extraction is changing the game.** GraphGen, structured LLM prompting, and other LLM-native approaches need systematic comparison against traditional IE — no framework exists for this.
- **KG quality is underexplored.** The community lacks consensus on what makes a KG "good." KGLab's evaluation suite is a proposal for standardization.
- **The field is fragmented.** AKBC, NLP, and Semantic Web communities all build KGs but don't share tools. A unifying library could bridge these communities.

## The PyG Parallel

| PyTorch Geometric (2019) | KGLab (proposed) |
|---|---|
| Standardized GNN operations | Standardized KG construction stages |
| CUDA kernels + mini-batching | Registry pattern + lazy imports |
| Comparative benchmarks across GNN architectures | Comparative benchmarks across extraction/resolution methods |
| Community adoption through extensibility | Same — swap in your own extractor/resolver/exporter |
| ICLR RLGM Workshop | Target: EMNLP/ACL Systems, AKBC, or JMLR MLOSS |

## The Core Paper Narrative

1. **Problem:** KG construction is fragmented; no standard library, no fair benchmarks, no shared evaluation.
2. **Solution:** KGLab — a modular, extensible library for end-to-end KG construction with built-in evaluation and benchmarking.
3. **Architecture:** The registry pattern, the tripartite model (entities/documents/chunks), the pipeline abstraction.
4. **Benchmarks:** Comparative results across extraction methods (spaCy vs. GraphGen vs. structured LLM), ontologies, and resolution strategies on a shared corpus.
5. **Case study:** KG-structured training data reduces LLM hallucination (QA generation → fine-tuning → evaluation).
6. **Impact:** Standardizing KG construction enables reproducible research and accelerates progress.

## What's Needed Before Submission

- [ ] Run the full benchmark matrix (extraction methods × ontologies × resolution strategies)
- [ ] Curate and release a standard evaluation corpus with gold KG annotations
- [ ] Add 1–2 more extraction baselines (e.g., OpenIE-style) for a richer comparison
- [ ] Produce the hallucination comparison rigorously (KG-trained vs. flat-trained LLM)
- [ ] Document the API with tutorials and examples (largely done)
- [ ] Package for PyPI with `pip install kglab`

## Target Venues (in priority order)

| Venue | Track | Rationale |
|---|---|---|
| **AKBC** | Main | Perfect topical fit — automated KB construction |
| **EMNLP / ACL** | System Demonstrations | Library paper + live demo |
| **NAACL / ACL** | Resources | Resource/library paper track |
| **JMLR** | MLOSS | Highest prestige for open-source ML software |
| **NeurIPS** | Datasets & Benchmarks | If the benchmark/eval angle is emphasized |
| **LREC / COLING** | Main | Language resources |
| **ISWC / ESWC** | Resources | Semantic web / knowledge graph community |

---

## Related Work to Cite

- **PyTorch Geometric** (Fey & Lenssen, 2019) — the model for this paper
- **DGL** (Wang et al., 2019) — another GNN library
- **GraphGen** — the joint extraction algorithm implemented in KGLab
- **DeepKE, OpenNRE, spaCy** — traditional IE tools
- **Neo4j, NetworkX** — graph backends
- **LLM-as-judge** (Zheng et al., 2023) — evaluation methodology
- **Hallucination in LLMs** — related to the KG-as-training-data case study
- **MinHash / LSH** — deduplication
- **SentenceTransformers** — embedding-based resolution
