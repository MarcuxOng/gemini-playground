from __future__ import annotations

from collections import OrderedDict

from limits import RateLimitItem
from limits import parse as parse_limit
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter

from app.config import settings

_storage = storage_from_string(settings.redis_url or "memory://")
_limiter = FixedWindowRateLimiter(_storage)
_cache: OrderedDict[str, RateLimitItem] = OrderedDict()
_MAX_CACHE_SIZE = 128


def check_tool_rate_limit(tool_name: str, limit_string: str) -> bool:
    """
    Return True if the call is within the rate limit.
    False if exceeded.
    """
    if tool_name not in _cache:
        if len(_cache) >= _MAX_CACHE_SIZE:
            _cache.popitem(last=False)
        _cache[tool_name] = parse_limit(limit_string)
    else:
        _cache.move_to_end(tool_name)
    return _limiter.hit(_cache[tool_name], "tool", tool_name)
