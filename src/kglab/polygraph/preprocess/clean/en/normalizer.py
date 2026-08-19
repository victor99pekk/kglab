"""English text normalization — language-specific cleaning for English documents."""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class EnglishCleaner:
    """English-specific text normalization."""

    def clean(self, text: str) -> str:
        text = text.strip()
        # Normalize Unicode — repair mojibake and canonicalize forms.
        text = unicodedata.normalize("NFC", text)
        try:
            from ftfy import fix_text

            text = fix_text(text)
        except ImportError:
            # ftfy is an optional dependency (install [curation] extra).
            # The base package remains functional without it.
            pass
        # Normalize line endings while preserving paragraph and line structure.
        # Quality filtering uses line boundaries to identify copied boilerplate.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Normalize dashes
        text = text.replace("\u2013", "-").replace("\u2014", "--")
        # Normalize ellipsis
        text = text.replace("\u2026", "...")
        # Strip non-printable characters (keep newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text
