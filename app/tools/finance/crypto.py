"""
Crypto Tool - A tool for fetching cryptocurrency prices using the CoinGecko API.
"""

from __future__ import annotations

import logging
import requests

from app.config import settings
from app.tools import register

logger = logging.getLogger(__name__)


def find_crypto_id(query: str) -> str | None:
    """
    Finds the CoinGecko ID for a given cryptocurrency symbol or name.
    """
    try:
        logger.info(f"Searching for CoinGecko ID for query: '{query}'")
        search_url = f"{settings.crypto_base_url}/search"
        response = requests.get(search_url, params={"query": query}, timeout=5)
        response.raise_for_status()
        data = response.json()        
        coins = data.get("coins", [])
        if coins:
            return str(coins[0].get("id")) if coins[0].get("id") else None
        return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API error during CoinGecko ID search: {e}")
        return None


@register
def get_crypto_price(query: str) -> str:
    """
    Get the current price of any cryptocurrency by its symbol or name from CoinGecko.
    :param query: The cryptocurrency symbol (e.g., 'BTC'), name ('Bitcoin'), or other identifier.
    """
    try:
        # Dynamically find the crypto ID from the user's query
        crypto_id = find_crypto_id(query)
        if not crypto_id:
            return f"Error: Could not find a matching cryptocurrency for '{query}'. Please try a different name or symbol."

        logger.info(f"Fetching crypto price for: {query} (resolved to ID: {crypto_id}) from CoinGecko")
        price_url = f"{settings.crypto_base_url}/simple/price"
        params = {
            "ids": crypto_id,
            "vs_currencies": "usd"
        }
        response = requests.get(
            price_url, 
            params=params, 
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        price = data.get(crypto_id, {}).get("usd")
        
        if price is not None:
            formatted_price = (f"{price:,.2f}" if price >= 1 else f"{price:,.8f}".rstrip("0").rstrip("."))
            return f"The current price of {query} ({crypto_id}) is ${formatted_price} USD."
        else:
            logger.warning(f"Could not find price for {crypto_id} in CoinGecko response: {data}")
            return f"Could not retrieve a valid price for {query}. The symbol might be incorrect or the API is busy."

    except Exception as e:
        logger.error(f"Error fetching crypto price: {e}")
        return f"Error fetching crypto price: {e}"