from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import gemini_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def map_search(query: str) -> str:
    """Search Google Maps for location-based information.

    :param query: The map search query (e.g., places, directions, nearby, etc.).
    """
    if not check_tool_rate_limit("map_search", "10/minute"):
        return "Rate limit exceeded: max 10 requests per minute."
    try:
        # Use gemini_service with map search grounding enabled
        return gemini_service(model=default_model, prompt=query, native_tools=["location"])
    except Exception as e:
        logger.error(f"Error in map_search tool: {e}")
        return "Map Search failed"
