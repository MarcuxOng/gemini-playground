from __future__ import annotations

import logging
import time

from limits import RateLimitItem
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

logger = logging.getLogger(__name__)

# failure posture when the rate-limit store is unreachable: degrade to per-instance 
# in-memory counters. Not fail-open, which would leave no ceiling on paid Gemini calls 
# during an outage; not a 503, which keeps the store's availability as a hard veto over 
# the surface Option B makes the product. The cost is that counters stop being shared, 
# so N instances allow N x the intended limit — a weaker guarantee, and only while the 
# primary store is down.
#
# There are three rate-limit call sites and they all run this posture. The HTTP routes
# get it from slowapi for free; the MCP middleware and the per-tool limiter both count
# on raw `limits` primitives, which have no such support, and are served by
# `limiter_hit` below so a third hand-rolled variant cannot drift in.
_PRIMARY_STORAGE_URI = settings.redis_url or "memory://"

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_PRIMARY_STORAGE_URI,
    # slowapi catches the storage error, logs a warning, flips to an in-memory limiter
    # and re-runs the check. It probes the primary periodically and recovers on its own.
    in_memory_fallback_enabled=True,
)


# --- Non-slowapi call sites --------------------------------------------------------
# `MCPAuthMiddleware` and `check_tool_rate_limit` both count with `limits` directly
# rather than through slowapi, so the fallback slowapi provides for free is mirrored
# by hand once, here, and shared by both.

_primary = FixedWindowRateLimiter(storage_from_string(_PRIMARY_STORAGE_URI))
_fallback = FixedWindowRateLimiter(storage_from_string("memory://"))

# How long to wait before re-probing a store already marked dead. Probing on every
# request would put the outage's latency back on the hot path we just removed it from.
_RECOVERY_PROBE_SECONDS = 30.0

_storage_dead = False
_last_probe = 0.0


def limiter_hit(limit: RateLimitItem, *identifiers: str) -> bool:
    """
    Count one request, degrading to in-memory counters if the store is down.

    Used by the two call sites that count with `limits` directly: the MCP middleware
    and the per-tool limiter. Returns ``True`` when the call is within its limit.

    A store outage never reaches the caller — it sees a normal allow/deny, counted
    per-instance until the primary answers again. Both callers need that guarantee for
    different reasons: the MCP middleware checks the limit *before* the API key, so an
    outage would otherwise close the surface at its first gate, and `check_tool_rate_limit`
    is called outside its caller's try/except, so a raised error would surface as an
    opaque tool failure.
    """
    global _storage_dead, _last_probe

    if _storage_dead and (time.monotonic() - _last_probe) >= _RECOVERY_PROBE_SECONDS:
        _last_probe = time.monotonic()
        try:
            if _primary.storage.check():
                logger.warning("Rate limit storage recovered — resuming shared counters.")
                _storage_dead = False
        except Exception:
            logger.debug("Rate limit storage still unreachable.", exc_info=True)

    if not _storage_dead:
        try:
            return _primary.hit(limit, *identifiers)
        except Exception:
            logger.warning(
                "Rate limit storage unreachable — falling back to in-memory counters. "
                "Limits are per-instance until it recovers.",
                exc_info=True,
            )
            _storage_dead = True
            _last_probe = time.monotonic()

    return _fallback.hit(limit, *identifiers)


def limiter_storage_status() -> dict[str, object]:
    """
    Report the rate-limit store's health, for `/api/v1/health` (T13).

    ``degraded`` is the field that matters. A degraded limiter is not an outage — the
    API is up and still limiting — but counters are per-instance, so it has to be
    visible somewhere other than a log line nobody reads.

    Only the URI scheme is reported, never the connection string, which carries
    credentials.
    """
    try:
        reachable = bool(_primary.storage.check())
    except Exception:
        reachable = False

    return {
        "backend": _PRIMARY_STORAGE_URI.split("://", 1)[0],
        "reachable": reachable,
        # slowapi tracks the HTTP surface's state privately; read it defensively so a
        # slowapi upgrade degrades this report rather than breaking the health check.
        "degraded": bool(getattr(limiter, "_storage_dead", False)) or _storage_dead,
    }
