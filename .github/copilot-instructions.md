# Polygraph — Agent Instructions

## Project Identity

Polygraph is both a **research project** and an **open-source library** for knowledge graph generation. All code must balance two goals:

1. **Research agility** — enable rapid experimentation and iteration on novel KG construction techniques.
2. **Library usability** — be intuitive for new users and easy to extend with new functionality.

When writing or modifying code, ask: *Would a new user understand how to use this? Could a contributor easily swap out this component for their own?*

## Code Style & Conventions

### Imports

- **All imports MUST be at the top of the file.** Never place import statements inside functions, methods, conditionals, or anywhere other than the module-level top of the file. This includes standard library, third-party, and local imports.
- Group imports in this order, separated by a blank line:
  1. Standard library imports
  2. Third-party imports
  3. Local (polygraph) imports

### General

- **Types must always be made clear.** Use type hints on all public function/method signatures, class attributes, and any non-obvious local variables. Never leave a reader guessing about what type something is.
- Prefer composition over inheritance for extensibility.
- Keep module-level APIs minimal — expose only what a user needs via `__all__` or explicit re-exports in `__init__.py`.

## Extensibility

- Core abstractions (extractors, resolvers, chunkers, exporters, etc.) should have clear, documented interfaces so users can plug in custom implementations.
- Avoid hardcoding provider-specific logic (e.g., a specific LLM or vector store). Use dependency injection or factory patterns.
- New pipelines, exporters, or extraction strategies should be addable without modifying existing core code — follow the plugin/registry pattern already established in the codebase.

---

<!-- Add your own project-specific instructions below this line -->
