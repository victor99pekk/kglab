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

Keep the design modular:

- new chunking, extraction, resolution, or export logic should fit into the existing stage interfaces
- prefer subclassing or swapping components instead of editing core logic in place
- keep new pipeline variants easy to register and compare

## Pull requests

1. Create a feature branch
2. Add or update tests
3. Run the relevant checks
4. Submit a PR with a clear summary of the change and why it matters

## Questions

If you are unsure where a change belongs, start by looking at the existing pipeline and stage modules in `kglab/` and the examples in the tutorial notebooks.
