"""
Token Counter tool — estimates token usage for a string.
"""

from __future__ import annotations

import logging

import tiktoken

from app.tools import register

logger = logging.getLogger(__name__)


@register
def count_tokens(text: str, model_encoding: str = "cl100k_base", max_tokens: int = 0) -> str:
    """
    Count the tokens for a given string.
    Use this to self-monitor context usage and prevent exceeding limits.

    :param text: The text to count tokens for.
    :param model_encoding: Encoding scheme to use (default cl100k_base, used by GPT-4 and O1).
    :param max_tokens: Optional limit; warn if count exceeds this (0 = no limit).
    """
    try:
        encoding = tiktoken.get_encoding(model_encoding)
        count = len(encoding.encode(text))

        result = f"Exact count: {count} tokens (Encoding: {model_encoding})."
        if max_tokens > 0 and count > max_tokens:
            result += f" WARNING: exceeds limit of {max_tokens} tokens."
        return result

    except Exception as e:
        logger.error(f"Token counter error: {e}")
        return f"Error counting tokens: {str(e)}"
