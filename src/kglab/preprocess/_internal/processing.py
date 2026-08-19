"""Backward-compatibility shim — imports moved to curate.en.processing."""

from kglab.preprocess.curate.en.processing import (
    DEFAULT_BGE_MODEL,
    BgeTokenCounter,
    CurationTextProcessor,
    SemanticReviewer,
    TextSpan,
    split_text_to_token_limit,
)

__all__ = [
    "BgeTokenCounter",
    "CurationTextProcessor",
    "DEFAULT_BGE_MODEL",
    "SemanticReviewer",
    "TextSpan",
    "split_text_to_token_limit",
]
