"""Incremental KG updates: add new documents to an existing knowledge graph.

Public API:
    from polygraph.kg_update import add_documents

Usage:
    add_documents(new_chunks, existing_kg_path, output_path)
"""


def add_documents(new_chunks, existing_kg_path, output_path, **kwargs):
    """Add new preprocessed chunks to an existing knowledge graph.

    This is a high-level wrapper that orchestrates extraction, resolution,
    and merging into the existing graph. For detailed control, use
    kg_build and kg_export modules directly.

    Args:
        new_chunks: Preprocessed Document chunks to add.
        existing_kg_path: Path to the existing KG JSON file.
        output_path: Where to write the merged KG.
        **kwargs: Passed through to extraction/resolution.
    """
    raise NotImplementedError(
        "kg_update.add_documents is a placeholder. Chain kg_build + manual merge for now."
    )
