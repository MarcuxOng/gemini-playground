from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import gemini_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def google_search(query: str) -> str:
    """Search Google for real-time information with citations.

    :param query: The search query.
    """
    if not check_tool_rate_limit("google_search", "10/minute"):
        return "Rate limit exceeded: max 10 search requests per minute."
    try:
        # Use gemini_service with search grounding enabled
        # This will return a response with citations formatted
        return gemini_service(model=default_model, prompt=query, native_tools=["search"])
    except Exception as e:
        logger.error(f"Error in google_search tool: {e}")
        return "Search failed"
