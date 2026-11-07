from __future__ import annotations

import logging

from app.config import default_model
from app.services.gemini import ask_file_service
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def gemini_ask_file(path: str, question: str) -> str:
    """
    Ask Gemini a question about a local file — PDF, image, audio or video — using native multimodal understanding. 
    No conversion step: the file is reasoned over in its original form.

    :param path: Path to the file on the machine running this server.
    :param question: What to ask about the file.
    """
    if not check_tool_rate_limit("gemini_ask_file", "10/minute"):
        return "Rate limit exceeded: max 10 file questions per minute."
    try:
        return ask_file_service(file_path=path, question=question, model=default_model)
    except FileNotFoundError as e:
        return str(e)
    except ValueError as e:
        return f"Unsupported or invalid file: {e}"
    except Exception as e:
        logger.error(f"Error in gemini_ask_file tool: {e}")
        return "File question failed"
