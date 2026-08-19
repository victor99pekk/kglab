"""Shared English relation-extraction parsing helpers."""

import re


def sentences(text: str) -> list[str]:
    """Split text into evidence-sized sentences while retaining punctuation."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if part.strip()]


def find_evidence(text: str, subject: str, object_: str) -> str:
    """Find the sentence containing both relation endpoints."""
    subject_key = subject.casefold()
    object_key = object_.casefold()
    for sentence in sentences(text):
        sentence_key = sentence.casefold()
        if subject_key in sentence_key and object_key in sentence_key:
            return sentence
    return ""
