# Contributing

Thanks for contributing to KGLab.

## Setup

```bash
git clone https://github.com/victor99pekk/kglab.git
cd kglab
uv sync
uv run python -m spacy download en_core_web_sm
uv run pytest tests/ -v
```

## Development workflow

- Use Python 3.10+
- Keep public APIs typed and documented
- Prefer small, focused changes
- Add or update tests for behavior changes

## Code quality

Run these before opening a PR:

```bash
uv run ruff check kglab/ ml/ tests/
uv run pytest tests/ -v
```

## Adding new pipeline stages or components

Follow this pattern when extending the project:

1. Add the new logic in the relevant pipeline stage-folders, in [kglab/preprocess](kglab/preprocess), [kglab/kg_build](kglab/kg_build), [kglab/kg_export](kglab/kg_export), (e.g. [kglab/kg_build/extract/entity](kglab/kg_build/extract/entity/) for new enitity extraction logic). If the new logic is language dependent add it in the language subfolder (e.g. [kglab/kg_build/extract/entity/en](kglab/kg_build/extract/entity/en/) for english)
2. Use the new functions in a new pipeline variant or override an existing pipeline step where appropriate, [kglab/pipelines](kglab/pipelines).
