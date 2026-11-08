from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import read_url_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def gemini_read_url(url: str, question: str) -> str:
    """
    Read a web page and answer a question grounded in its content, in one call.

    The page is fetched and reasoned over by Gemini natively (url_context) —
    no separate fetch step — and the answer carries the read pages as sources.

    :param url: The page to read.
    :param question: What to ask about the page.
    """
    if not check_tool_rate_limit("gemini_read_url", "10/minute"):
        return "Rate limit exceeded: max 10 URL reads per minute."
    try:
        return read_url_service(url=url, question=question, model=default_model)
    except Exception as e:
        logger.error(f"Error in gemini_read_url tool: {e}")
        return "URL read failed"
