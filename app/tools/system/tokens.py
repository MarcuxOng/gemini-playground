"""
Token Counter tool — estimates token usage for a string.
"""

from __future__ import annotations

import logging
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

from app.tools import register

logger = logging.getLogger(__name__)


@register
def count_tokens(text: str, model_encoding: str = "cl100k_base") -> str:
    """
    Count the tokens for a given string.
    Use this to self-monitor context usage and prevent exceeding limits.

    :param text: The text to count tokens for.
    :param model_encoding: Encoding scheme to use (default cl100k_base, used by GPT-4 and O1).
    """
    try:
        if not _HAS_TIKTOKEN:
            # Simple fallback estimation (approx. 4 chars per token)
            count = len(text) // 4
            return f"Estimate: {count} tokens (Tiktoken not found, using simple estimation)."

        encoding = tiktoken.get_encoding(model_encoding)
        count = len(encoding.encode(text))
        
        return f"Exact count: {count} tokens (Encoding: {model_encoding})."

    except Exception as e:
        logger.error(f"Token counter error: {e}")
        return f"Error counting tokens: {str(e)}"
