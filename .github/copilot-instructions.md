# KGLab — Agent Instructions

## Project Identity

KGLab is both a **research project** and an **open-source library** for knowledge graph generation. All code must balance two goals:

1. **Research agility** — enable rapid experimentation and iteration on novel KG construction techniques. The knowldege graph will only ever support three types of nodes: entities, documents, and chunks, for now at least.
  - user should be able to quickly swap out components (e.g., chunkers, extractors, resolvers) and run experiments with minimal boilerplate.
  - users should be able to easily compare results across different pipelines and configurations.
  - users should be able to easily add new pipelines, exporters, or extraction strategies without modifying existing core code.
  - there should be good support for knowledge graph generation pipelines, knowledge graph export, and evaluation metrics.
2. **Library usability** — be intuitive for new users and easy to extend with new functionality. It is extremely important that it is intutitive for users to create new pipelines and use existing pipelines with python code. The library should be easy to use for new users, and easy to extend with new functionality. It should be easy to create new pipelines and use existing pipelines with python code.

3. **Code quality** — maintain a high standard of code readability, maintainability, and test coverage.
4. The codebase should be well-documented, with clear docstrings and usage examples for all public APIs.
5. I want the codebase to be like a libary for KG operations, with clear abstractions and interfaces for each component (e.g., chunkers, extractors, resolvers, exporters). But in my specific use case i am building this to create a Great Knowledge Graph for LLM training.

When writing or modifying code, ask: *Would a new user understand how to use this? Could a contributor easily swap out this component for their own?*

## Code Style & Conventions

### Imports

- **All imports MUST be at the top of the file.** Never place import statements inside functions, methods, conditionals, or anywhere other than the module-level top of the file. This includes standard library, third-party, and local imports.
- Group imports in this order, separated by a blank line:
  1. Standard library imports
  2. Third-party imports
  3. Local (kglab) imports

### General

- **Types must always be made clear.** Use type hints on all public function/method signatures, class attributes, and any non-obvious local variables. Never leave a reader guessing about what type something is.
- Prefer composition over inheritance for extensibility.
- Keep module-level APIs minimal — expose only what a user needs via `__all__` or explicit re-exports in `__init__.py`.

## Extensibility

- Core abstractions (extractors, resolvers, chunkers, exporters, etc.) should have clear, documented interfaces so users can plug in custom implementations.
- Avoid hardcoding provider-specific logic (e.g., a specific LLM or vector store). Use dependency injection or factory patterns.
- New pipelines, exporters, or extraction strategies should be addable without modifying existing core code — follow the plugin/registry pattern already established in the codebase.

---
