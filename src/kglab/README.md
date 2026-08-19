# KG Generator Architecture

This package has two connected workflows.

```text
Raw documents → curate → curated dataset + audit
Curated documents → extract → resolve → graph → evaluate/export
```

## Curation workflow

`ingest/` loads files. `dedup/` profiles quality and finds repeated content. `curate/` performs language-aware normalization, layered duplicate decisions, sentence-safe token records, semantic review, deterministic shards, and provenance audits.

## Knowledge Graph workflow

`kg_build/extract/` finds entities and relationships. `kg_build/resolve/`
merges references to the same entity. `kg_build/build/` creates the KG.
`kg_eval/` measures its quality, and `kg_export/` writes files for downstream
tools.

Each selectable strategy has one implementation module:

```text
kg_build/
├── extract/
│   ├── entity/
│   │   ├── spacy.py
│   │   └── regex.py
│   ├── relation/
│   │   ├── ontology_rules.py
│   │   └── structured_llm.py
│   ├── joint/
│   │   └── graphgen.py       # GraphGen implementation and its prompts
│   └── registry.py
├── resolve/
│   ├── string.py
│   ├── embedding.py
│   └── registry.py
└── build/
    ├── networkx.py
    └── registry.py
```

Composed extraction selects one entity method and one relation method. Joint
methods, such as GraphGen, own both tasks and are selected separately.

## Important distinction

- `dedup/` works on **documents** before extraction.
- `resolve/` works on **entities** after extraction.

Import implementations from their method packages, or select them through the
stage registries.
