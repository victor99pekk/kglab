"""Language-specific QA templates and patterns for dataset generation.

Shared by all QA generator implementations. Zero dependencies — pure data.
"""

# ── KG QA templates ────────────────────────────────────────────
# Placeholders: {subj}, {rel}, {obj}

_KG_SINGLE_HOP_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("What is the {rel} of {subj}?", "{subj} {rel} {obj}."),
        ("{subj} {rel} what?", "{obj}."),
        ("Who or what does {subj} {rel}?", "{subj} {rel} {obj}."),
    ],
}

_KG_MULTI_HOP_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "Starting from {start}, follow these relationships in order: {relations}. Which entity is reached?",
        "{answer}",
    ),
}

_KG_COMPARISON_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "Compare {e1} and {e2}. What do they have in common?",
        "{answer}",
    ),
}

_KG_TRUE_FALSE_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": (
        "True or False: {subj} {rel} {obj}.",
        "{answer}",
    ),
}

# ── Raw text QA templates ──────────────────────────────────────

_RAW_DEFINITION_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": ("What is {subj}?", "{pred}"),
}

_RAW_RELATIONSHIP_TEMPLATES: dict[str, tuple[str, str]] = {
    "en": ("What is the relationship between {e1} and {e2}?", "{sent}"),
}

# ── Regex patterns for raw-text extraction ─────────────────────
# Each: (pattern, question_template)
#   pattern: regex with two capture groups (subject, object)
#   question_template: {0}=subject placeholder

_RAW_FACT_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "en": [
        (r"(.+?)\s+(?:was\s+)?born\s+in\s+(.+?)(?:,|\.|$)", "Where was {0} born?"),
        (r"(.+?)\s+(?:worked|works)\s+(?:at|for|in)\s+(.+?)(?:,|\.|$)", "Where did {0} work?"),
        (r"(.+?)\s+(?:studied|studies)\s+(?:at|in)\s+(.+?)(?:,|\.|$)", "Where did {0} study?"),
        (r"(.+?)\s+(?:died|dies)\s+in\s+(.+?)(?:,|\.|$)", "Where did {0} die?"),
        (
            r"(.+?)\s+(?:discovered|invented|created|developed)\s+(.+?)(?:,|\.|$)",
            "What did {0} discover?",
        ),
    ],
}

# ── Copula / definition patterns ───────────────────────────────

_COPULA_PATTERNS: dict[str, str] = {
    "en": r"(.+?)\s+(is|was|are|were)\s+(a|an|the)?\s*(.+)",
}

_PRONOUNS: dict[str, set[str]] = {
    "en": {"he", "she", "his", "her", "they", "their", "it", "its", "this", "that"},
}

_STRUCTURAL_PREDICATES = {"NEXT", "PART_OF", "MENTIONS"}
