from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import grounded_answer_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def gemini_grounded_answer(question: str) -> str:
    """
    Answer a question grounded in real web search, with the full citation map.

    Goes beyond a plain source list: alongside the answer you get the Sources
    list AND a Citations block mapping which parts of the answer are backed by
    which sources — Gemini's own segment-to-source grounding supports.

    :param question: The question to research and answer.
    """
    if not check_tool_rate_limit("gemini_grounded_answer", "10/minute"):
        return "Rate limit exceeded: max 10 grounded answers per minute."
    try:
        return grounded_answer_service(question=question, model=default_model)
    except Exception as e:
        logger.error(f"Error in gemini_grounded_answer tool: {e}")
        return "Grounded answer failed"
