from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import maps_answer_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def gemini_maps_answer(question: str) -> str:
    """
    Answer location questions with Google Maps grounding — real Places data, not hallucinated names.
    Returns place names, addresses, ratings, opening hours with their Google Maps links.

    :param question: The location question, e.g. "cafes near Raffles Place with good coffee".
    """
    if not check_tool_rate_limit("gemini_maps_answer", "10/minute"):
        return "Rate limit exceeded: max 10 maps questions per minute."
    try:
        return maps_answer_service(question=question, model=default_model)
    except Exception as e:
        logger.error(f"Error in gemini_maps_answer tool: {e}")
        return "Maps answer failed"
