"""
Stocks Tools - A collection of tools for stock market analysis and trading strategies.
"""

from __future__ import annotations

import logging

import requests

from app.config import settings
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a given symbol.
    :param symbol: The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    """
    if not check_tool_rate_limit("get_stock_price", "5/minute"):
        return "Rate limit exceeded: max 5 stock price requests per minute."
    try:
        logger.info(f"Fetching stock price for: {symbol}")
        url = settings.alpha_vantage_base_url
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": settings.alpha_vantage_api_key,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Check for API error messages
        if "Error Message" in data:
            error_message = data["Error Message"]
            logger.error(f"Alpha Vantage API error: {error_message}")
            return f"Error from Alpha Vantage: {error_message}"

        quote = data.get("Global Quote", {})
        price = quote.get("05. price")
        if price:
            return f"The current price of {symbol} is ${price}."
        else:
            # Provide more context on failure
            logger.warning(
                f"Could not find 'Global Quote' or '05. price' in response for {symbol}. Full response: {data}"
            )
            return f"Could not retrieve a valid price for {symbol}. The symbol might be incorrect or the API limit reached."

    except requests.RequestException as e:
        logger.error(f"Error fetching stock price: {e}")
        return f"Error fetching stock price: {e}"
    except Exception as e:
        logger.error(f"Unknown error fetching stock price: {e}")
        return f"Error fetching stock price: {e}"
