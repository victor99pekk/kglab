"""Sentence-aware, token-budgeted chunking.

Exports: SentenceChunker
"""

import re
from collections.abc import Sequence

from polygraph._shared import Document, Language
from polygraph.preprocess.chunk.fixed import TextChunker, count_tokens


def _sentence_units(text: str, language: Language) -> list[str]:
    """Split readable text into sentences without exposing tokenizer underscores."""
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+|[\r\n]+", text) if part.strip()]


class SentenceChunker(TextChunker):
    """Pack complete sentences into token-budgeted, language-aware chunks."""

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 60,
        language: Language = Language.ENGLISH,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be greater than zero")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be between zero and target_tokens")
        super().__init__(chunk_size=target_tokens, chunk_overlap=overlap_tokens)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.language = language

    def chunk_count(self, text: str) -> int:
        probe = Document(content=text)
        return len(self._chunk_one(probe))

    def _chunk_one(self, doc: Document) -> list[Document]:
        units = self._expanded_units(doc.content)
        if not units:
            return []
        packed = self._pack_units(units)
        if len(packed) == 1:
            doc.metadata["chunk_index"] = 0
            doc.metadata["token_count"] = count_tokens(doc.content)
            return [doc]
        return [self._make_chunk(doc, text, index) for index, text in enumerate(packed)]

    def _expanded_units(self, text: str) -> list[str]:
        expanded: list[str] = []
        for sentence in _sentence_units(text, self.language):
            words = sentence.split()
            if len(words) <= self.target_tokens:
                expanded.append(sentence)
                continue
            start = 0
            while start < len(words):
                expanded.append(" ".join(words[start : start + self.target_tokens]))
                start += self.target_tokens
        return expanded

    def _pack_units(self, units: Sequence[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            unit_tokens = count_tokens(unit)
            if current and current_tokens + unit_tokens > self.target_tokens:
                chunks.append(" ".join(current))
                current = self._overlap_units(current)
                current_tokens = sum(count_tokens(item) for item in current)
                if current and current_tokens + unit_tokens > self.target_tokens:
                    current = []
                    current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _overlap_units(self, units: Sequence[str]) -> list[str]:
        if not self.overlap_tokens:
            return []
        selected: list[str] = []
        tokens = 0
        for unit in reversed(units):
            unit_tokens = count_tokens(unit)
            if tokens + unit_tokens > self.overlap_tokens:
                break
            selected.append(unit)
            tokens += unit_tokens
        return list(reversed(selected))
