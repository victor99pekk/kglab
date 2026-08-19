"""Abstract preprocessing contract — swap the entire pipeline or compose stages.

Provides the ``Preprocessor`` ABC that pipelines depend on. Subclass to
replace the entire preprocessing strategy, or use ``DefaultPreprocessor``
with ``PreprocessConfig`` for stage-level customization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from kglab._shared import Document


class Preprocessor(ABC):
    """Raw input paths → clean, deduplicated Document chunks.

    The pipeline calls ``preprocess()`` once during execution and
    may later access ``raw_docs`` for document-to-document relation
    extraction.  Implementations are not required to retain raw
    documents — if ``raw_docs`` returns an empty list downstream
    stages that need them will be skipped.

    Subclassing guide::

        class MyPreprocessor(Preprocessor):
            @property
            def supported_languages(self) -> set[str]:
                return {"en", "de"}

            def preprocess(self, input_paths: list[Path]) -> list[Document]:
                docs = load.from_paths(input_paths)
                docs = clean.normalize(docs)
                chunks = chunk.by_sentence(docs)
                self._raw = docs
                return chunks

            @property
            def raw_docs(self) -> list[Document]:
                return getattr(self, "_raw", [])
    """

    @abstractmethod
    def preprocess(self, input_paths: list[Path]) -> list[Document]:
        """Load, clean, chunk, and deduplicate documents from input paths.

        Returns:
            Clean, deduplicated ``Document`` chunks ready for extraction.
        """
        ...

    @property
    def supported_languages(self) -> set[str]:
        """Languages this preprocessor can handle.

        ``"*"`` means universal (no language restriction).  Only
        documents whose detected language is in this set will be
        processed; others are skipped with a log message.
        """
        return {"*"}

    @property
    def raw_docs(self) -> list[Document]:
        """Raw documents retained for downstream relation extraction.

        The default returns an empty list.  Override this property
        if your preprocessor stores raw documents that downstream
        stages (e.g. document-to-document relation extraction) need.

        .. note::
           Retaining raw documents *and* chunks simultaneously
           roughly doubles memory.  If memory is a concern, return
           an empty list here to skip document-relation extraction.
        """
        return []
