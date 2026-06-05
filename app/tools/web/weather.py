"""
Weather tool — provides the LLM with current weather information via OpenWeatherMap.
"""

from __future__ import annotations

import logging

import requests

from app.config import settings
from app.tools import register
from app.utils.tool_limiter import check_tool_rate_limit

logger = logging.getLogger(__name__)


@register
def get_weather(location: str, units: str = "metric") -> str:
    """
    Get the current weather for a given location (city, country, or lat/lon).
    Use this when the user asks for weather, temperature, or climate conditions.

    :param location: The location to get weather for (e.g., 'London, UK' or '1.35,103.8').
    :param units: Unit system for temperature ('metric', 'imperial', or 'standard'). Defaults to 'metric'.
    """
    if not check_tool_rate_limit("get_weather", "30/minute"):
        return "Rate limit exceeded: max 30 weather requests per minute."
    try:
        logger.info(f"Fetching weather for: {location} (units: {units})")

        params = {
            "appid": settings.openweathermap_api_key,
            "units": units,
        }

        # Handle lat,lon coordinates vs city name
        if "," in location and all(
            part.strip().replace(".", "", 1).replace("-", "", 1).isdigit()
            for part in location.split(",")
        ):
            lat, lon = location.split(",")
            params["lat"] = lat.strip()
            params["lon"] = lon.strip()
        else:
            params["q"] = location

        response = requests.get(settings.weather_base_url, params=params, timeout=10)

        if response.status_code == 401:
            return "Error: Invalid OpenWeatherMap API key."
        if response.status_code == 404:
            return f"Error: Location '{location}' not found."
        if not response.ok:
            return f"Error: API returned status {response.status_code}."

        data = response.json()

        # Parse into a human-readable summary
        city = data.get("name", "Unknown")
        country = data.get("sys", {}).get("country", "")
        temp = data.get("main", {}).get("temp")
        feels_like = data.get("main", {}).get("feels_like")
        desc = data.get("weather", [{}])[0].get("description", "clear sky").capitalize()
        humidity = data.get("main", {}).get("humidity")
        wind_speed = data.get("wind", {}).get("speed")

        unit_symbol = "°C" if units == "metric" else "°F" if units == "imperial" else "K"
        speed_unit = "m/s" if units != "imperial" else "mph"

        summary = (
            f"Weather in {city}, {country}:\n"
            f"- Condition: {desc}\n"
            f"- Temperature: {temp}{unit_symbol} (Feels like {feels_like}{unit_symbol})\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind Speed: {wind_speed} {speed_unit}"
        )

        return summary

    except Exception as e:
        logger.error(f"Weather tool error: {e}")
        return f"Error fetching weather: {str(e)}"
