from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import run_code_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def gemini_run_code(task: str) -> str:
    """
    Run a computation task in Gemini's sandboxed Python environment.

    The model writes and executes Python in Google's sandbox (never on this
    machine) — math, data crunching, format conversion — and returns the
    answer along with the code it ran and the execution output.

    :param task: The computation to perform, in natural language.
    """
    if not check_tool_rate_limit("gemini_run_code", "10/minute"):
        return "Rate limit exceeded: max 10 code runs per minute."
    try:
        return run_code_service(task=task, model=default_model)
    except Exception as e:
        logger.error(f"Error in gemini_run_code tool: {e}")
        return "Code execution failed"
