"""
News tool — fetches current news articles using NewsAPI.
"""

from __future__ import annotations

import logging

import requests

from app.config import settings
from app.tools import register

logger = logging.getLogger(__name__)


# DEPRECATED — prefer native_tools=["search"] in Phase 4.2
@register
def get_news(query: str, page_size: int = 5, language: str = "en") -> str:
    """
    Search for recent news articles and headlines on a specific topic.
    Use this to get current events, news, or recent developments.

    :param query: The topic or keywords to search for (e.g., 'Artificial Intelligence').
    :param page_size: Number of articles to return (default is 5).
    :param language: The language of the articles (default is 'en').
    """

    try:
        logger.info(f"Fetching news for query: {query}")
        url = settings.news_base_url
        params: dict[str, str | int] = {
            "q": query,
            "pageSize": page_size,
            "language": language,
            "sortBy": "publishedAt",
            "apiKey": settings.news_api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 401:
            return "Error: Invalid NewsAPI key."
        if not response.ok:
            return f"Error: NewsAPI returned status {response.status_code}."

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return f"No news articles found for '{query}'."

        results = [f"--- Top News for '{query}' ---"]
        for i, art in enumerate(articles, 1):
            title = art.get("title", "No Title")
            source = art.get("source", {}).get("name", "Unknown Source")
            description = art.get("description", "No description available.")
            url = art.get("url", "No URL")

            results.append(
                f"{i}. {title}\n   Source: {source}\n   Summary: {description}\n   Link: {url}"
            )

        return "\n\n".join(results)

    except Exception as e:
        logger.error(f"News tool error: {e}")
        return f"Error fetching news: {str(e)}"
