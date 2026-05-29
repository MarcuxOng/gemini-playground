"""
Wikipedia tool — fetches summaries and facts from Wikipedia.
"""

from __future__ import annotations

import logging

import requests

from app.config import settings
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def get_wikipedia_summary(query: str) -> str:
    """
    Fetch a clean summary of a topic from Wikipedia's API.
    Use this for facts, history, or detailed descriptions of people, places, or concepts.

    :param query: The search term or topic (e.g., 'Quantum mechanics').
    """
    if not check_tool_rate_limit("get_wikipedia_summary", "20/minute"):
        return "Rate limit exceeded: max 20 Wikipedia requests per minute."
    try:
        logger.info(f"Fetching Wikipedia summary for: {query}")
        search_url = f"{settings.wikipedia_base_url}"
        headers = {
            "User-Agent": "Gemini-Playground/1.0 (https://github.com/your-repo; mailto:your-email@example.com) Requests/2.31.0"
        }
        search_params: dict[str, str | int] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }

        search_res = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        search_res.raise_for_status()
        search_data = search_res.json()

        if not search_data.get("query", {}).get("search"):
            return f"No Wikipedia page found for '{query}'."

        page_title = search_data["query"]["search"][0]["title"]
        extract_params: dict[str, str | int | bool] = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": page_title,
            "format": "json",
        }
        extract_res = requests.get(search_url, params=extract_params, headers=headers, timeout=10)
        extract_res.raise_for_status()
        extract_data = extract_res.json()

        pages = extract_data.get("query", {}).get("pages", {})
        if not pages:
            return f"Could not retrieve summary for '{page_title}'."

        # The key is the page ID (string)
        page_id = next(iter(pages))
        extract = pages[page_id].get("extract", "")

        if not extract:
            return f"No summary available for '{page_title}'."

        return f"--- {page_title} (Wikipedia) ---\n{extract}"

    except Exception as e:
        logger.error(f"Wikipedia tool error: {e}")
        return f"Error fetching Wikipedia summary: {str(e)}"
