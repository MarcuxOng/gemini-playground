from __future__ import annotations

import logging

from app.services.gemini import gemini_service
from app.tools import register

logger = logging.getLogger(__name__)


@register
def google_search(query: str) -> str:
    """Search Google for real-time information with citations.

    :param query: The search query.
    """
    try:
        # Use gemini_service with search grounding enabled
        # This will return a response with citations formatted
        return gemini_service(model="gemini-2.5-flash", prompt=query, native_tools=["search"])
    except Exception as e:
        logger.error(f"Error in google_search tool: {e}")
        return f"Error performing search: {str(e)}"
