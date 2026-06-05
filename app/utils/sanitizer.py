from __future__ import annotations

import re
import unicodedata


class InputSanitizationError(Exception):
    pass


# High-confidence jailbreak patterns — chosen for precision over recall to avoid false positives on legitimate creative or instructional prompts.
_JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(
        r"(reveal|show|print|display|output)\s+(your\s+)?(system\s+prompt|instructions?)",
        re.IGNORECASE,
    ),
    re.compile(r"bypass\s+(your\s+)?(safety|content|filters?|guardrails?)", re.IGNORECASE),
    re.compile(r"\b(DAN|jailbreak)\s+mode\b", re.IGNORECASE),
    re.compile(r"(enable|activate)\s+(developer|god)\s+mode", re.IGNORECASE),
    re.compile(
        r"forget\s+(all\s+)?(your\s+)?(previous\s+)?(rules?|instructions?|guidelines?)",
        re.IGNORECASE,
    ),
]

# Control characters (excluding \t \n \r), zero-width chars, bidi overrides, bidi isolates, line/paragraph separators, and Unicode tag characters.
# These are commonly used in prompt injection and invisible-text attacks.
_DANGEROUS_UNICODE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    "​-‏"  # zero-width chars
    "‪-‮"  # bidi overrides (LRE, RLE, PDF, LRO, RLO)
    "⁦-⁩"  # bidi isolates (LRI, RLI, FSI, PDI)
    "\u2028\u2029"  # line/paragraph separators
    "\U000e0000-\U000e007f]"  # Unicode tags (invisible text injection)
)


def sanitize_prompt(text: str) -> str:
    """Strip dangerous characters and reject known jailbreak patterns.

    Returns the cleaned text, or raises InputSanitizationError if the input
    contains a disallowed pattern.
    """
    # Normalize Unicode to prevent encoding tricks (e.g. lookalike characters)
    text = unicodedata.normalize("NFC", text)

    # Strip control characters and invisible Unicode injections
    text = _DANGEROUS_UNICODE.sub("", text)

    # Reject high-confidence jailbreak attempts
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(text):
            raise InputSanitizationError("Input contains disallowed content")

    return text
