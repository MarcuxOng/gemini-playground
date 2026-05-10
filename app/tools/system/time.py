"""
Time tool — provides current date and time information.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    # Fallback for Python < 3.9
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools import register

logger = logging.getLogger(__name__)


@register
def get_datetime_info(timezone_name: str | None = None) -> str:
    """
    Get the current date and time, optionally in a specific timezone.
    
    :param timezone_name: IANA timezone (e.g., 'Asia/Singapore', 'America/New_York'). Defaults to UTC.
    """
    try:
        logger.info(f"Fetching datetime info for timezone: {timezone_name or 'UTC'}")
        tz = ZoneInfo(timezone_name) if timezone_name else UTC
        now = datetime.now(tz)
        
        return (
            f"Timezone: {timezone_name or 'UTC'}\n"
            f"Date: {now.strftime('%Y-%m-%d')}\n"
            f"Time: {now.strftime('%H:%M:%S')}\n"
            f"Weekday: {now.strftime('%A')}\n"
            f"Week Number: {now.isocalendar().week}\n"
            f"UTC Offset: {now.strftime('%z')}"
        )
    except ZoneInfoNotFoundError:
        return f"Error: Unknown timezone: '{timezone_name}'."
    except Exception as e:
        logger.error(f"Datetime tool error: {e}")
        return f"Error: {str(e)}"
