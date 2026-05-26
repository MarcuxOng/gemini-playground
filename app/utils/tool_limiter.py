from __future__ import annotations

from limits import RateLimitItem
from limits import parse as parse_limit
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter

from app.config import settings

_storage = storage_from_string(settings.redis_url or "memory://")
_limiter = FixedWindowRateLimiter(_storage)
_cache: dict[str, RateLimitItem] = {}


def check_tool_rate_limit(tool_name: str, limit_string: str) -> bool:
    """
    Return True if the call is within the rate limit.
    False if exceeded.
    """
    if tool_name not in _cache:
        _cache[tool_name] = parse_limit(limit_string)
    return _limiter.hit(_cache[tool_name], "tool", tool_name)
