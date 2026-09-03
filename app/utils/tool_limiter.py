from __future__ import annotations

from collections import OrderedDict

from limits import RateLimitItem
from limits import parse as parse_limit

from app.utils.limiter import limiter_hit

_cache: OrderedDict[str, RateLimitItem] = OrderedDict()
_MAX_CACHE_SIZE = 128


def check_tool_rate_limit(tool_name: str, limit_string: str) -> bool:
    """
    Return True if the call is within the rate limit, False if exceeded.

    Counting goes through `limiter_hit`, so an unreachable store degrades to
    per-instance counters rather than raising (T1 / Decision 1). That matters here
    because callers invoke this *outside* their own try/except — see
    `app/tools/web/search.py` — so a raised error would escape as an opaque tool
    failure rather than a rate-limit verdict.
    """
    if tool_name not in _cache:
        if len(_cache) >= _MAX_CACHE_SIZE:
            _cache.popitem(last=False)
        _cache[tool_name] = parse_limit(limit_string)
    else:
        _cache.move_to_end(tool_name)
    return limiter_hit(_cache[tool_name], "tool", tool_name)
