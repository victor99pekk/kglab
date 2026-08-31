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

1. Add the new logic to the relevant stage folder, e.g. [kglab/preprocess](kglab/preprocess), [kglab/kg_build](kglab/kg_build), [kglab/kg_export](kglab/kg_export), or [kglab/pipelines](kglab/pipelines).
2. Keep the implementation modular and aligned with the existing stage interfaces instead of editing core logic in place.
3. Use the new functions in a new pipeline variant or override an existing pipeline step where appropriate.

## Pull requests

1. Create a feature branch
2. Add or update tests
3. Run the relevant checks
4. Submit a PR with a clear summary of the change and why it matters
