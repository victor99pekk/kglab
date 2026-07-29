# Experiment 001 — Baseline Pipeline

**Date:** 2026-07-29
**Pipeline variant:** `baseline`
**Status:** planned

---

## Hypothesis

The default baseline pipeline (spaCy entity extraction + string-similarity entity resolution) produces a knowledge graph with reasonable completeness and consistency on general-domain text. This establishes a lower bound for quality that future experiments (LLM extraction, embedding resolution, etc.) should improve upon.

## Setup

- **Input data:** `data/wikipedia/`
- **Ontology:** `configs/default_ontology.yaml`
- **Pipeline:** spaCy `en_core_web_sm` for extraction, minhash dedup, string-based entity resolution (threshold 0.85)
- **Special parameters:** None — all defaults

## Results

| Metric | Value |
|---|---|
| Nodes | — |
| Edges | — |
| Triples | — |
| Completeness | — |
| Consistency | — |
| Duplication | — |
| Overall Score | — |
| Wall Time | — |

*See `outputs/results_summary.json` for the full report.*

## Conclusions

*To be filled after running.*
