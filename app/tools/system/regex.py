"""
Regex Tester tool — validates and tests regular expressions.
"""

from __future__ import annotations

import logging
import re

from app.tools import register

logger = logging.getLogger(__name__)


@register
def test_regex(pattern: str, text: str) -> str:
    r"""
    Test a regular expression against a string. 
    Returns all matches, groups, and positions.
    Use this to debug or verify regex patterns.

    :param pattern: The regex pattern (e.g., r'\d+').
    :param text: The text to search within.
    """
    try:
        logger.info(f"Testing regex pattern: {pattern}")
        matches = []
        for match in re.finditer(pattern, text):
            # Capture both full match and named/numbered groups
            matches.append({
                "match": match.group(0),
                "span": match.span(),
                "groups": match.groups(),
                "named_groups": match.groupdict()
            })

        if not matches:
            return f"No matches found for pattern: {pattern}"

        # Format output
        summary = [f"Found {len(matches)} match(es):"]
        for i, m in enumerate(matches, 1):
            line = f"{i}. '{m['match']}' at positions {m['span']}"
            if any(m['groups']):
                line += f" | Groups: {m['groups']}"
            if m['named_groups']:
                line += f" | Named Groups: {m['named_groups']}"
            summary.append(line)

        return "\n".join(summary)

    except re.error as e:
        return f"Regex Error: {str(e)}"
    except Exception as e:
        logger.error(f"Regex tool error: {e}")
        return f"Error during regex testing: {str(e)}"
