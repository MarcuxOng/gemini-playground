"""Rate limiter failure posture — T1 / Decision 1.

Decision 1 chose to degrade to per-instance in-memory counters when the store is
unreachable, rather than failing open or returning 503. These cover the
unreachable-store path, which previously surfaced as a 500 on every rate-limited
route and is the reason the server had to be started with `REDIS_URL=memory://`.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from limits import parse

from app.utils import limiter as limiter_module
from app.utils.limiter import limiter, limiter_hit, limiter_storage_status
from app.utils.tool_limiter import check_tool_rate_limit


@pytest.fixture(autouse=True)
def reset_limiter_state():
    """Degradation is module state; leaking it would make these order-dependent."""
    limiter_module._storage_dead = False
    limiter_module._last_probe = 0.0
    yield
    limiter_module._storage_dead = False
    limiter_module._last_probe = 0.0


def test_http_limiter_degrades_rather_than_failing():
    """The HTTP surface takes the chosen posture, not either rejected one."""
    assert limiter._in_memory_fallback_enabled is True
    # Fail-open was explicitly not chosen: it leaves no ceiling on paid Gemini calls.
    assert limiter._swallow_errors is False


def test_hit_allows_request_on_healthy_store():
    assert limiter_hit(parse("60/minute"), "mcp", "healthy-client") is True


def test_hit_does_not_raise_when_store_unreachable():
    """The regression under test: an outage became a 500 instead of an allow/deny."""
    with patch.object(limiter_module._primary, "hit", side_effect=ConnectionError("down")):
        allowed = limiter_hit(parse("60/minute"), "mcp", "unreachable-client")

    assert allowed is True
    assert limiter_module._storage_dead is True


def test_still_limits_while_degraded():
    """Degraded is not unlimited — this is what rules out fail-open."""
    limit = parse("2/minute")

    with patch.object(limiter_module._primary, "hit", side_effect=ConnectionError("down")):
        first = limiter_hit(limit, "mcp", "degraded-client")
        second = limiter_hit(limit, "mcp", "degraded-client")
        third = limiter_hit(limit, "mcp", "degraded-client")

    assert (first, second) == (True, True)
    assert third is False


def test_recovers_when_store_returns():
    """A recovered store resumes shared counters without a restart."""
    limiter_module._storage_dead = True
    limiter_module._last_probe = 0.0  # forces a probe on the next call

    with patch.object(limiter_module._primary.storage, "check", return_value=True):
        limiter_hit(parse("60/minute"), "mcp", "recovering-client")

    assert limiter_module._storage_dead is False


def test_does_not_probe_the_dead_store_on_every_request():
    """Probing per request would put the outage's latency back on the hot path."""
    limiter_module._storage_dead = True
    limiter_module._last_probe = 0.0

    with patch.object(limiter_module._primary.storage, "check", return_value=False) as probe:
        for _ in range(5):
            limiter_hit(parse("60/minute"), "mcp", "probe-client")

    assert probe.call_count == 1


def test_status_reports_degradation_for_the_health_check():
    """T13 needs this visible; a degraded limiter is silent otherwise."""
    with patch.object(limiter_module._primary, "hit", side_effect=ConnectionError("down")):
        limiter_hit(parse("60/minute"), "mcp", "status-client")

    assert limiter_storage_status()["degraded"] is True


def test_status_never_leaks_the_connection_string():
    """The store URI carries credentials in production; only the scheme is reported."""
    status = limiter_storage_status()

    assert status["backend"] == "memory"
    assert "://" not in str(status["backend"])


def test_tool_limiter_does_not_raise_when_store_unreachable():
    """The third call site, and the one with the nastiest failure mode.

    `check_tool_rate_limit` is invoked *outside* its caller's try/except — see
    `app/tools/web/search.py` — so before this it escaped as an opaque tool error
    rather than a rate-limit verdict. Every Option B tool goes through this path.
    """
    with patch.object(limiter_module._primary, "hit", side_effect=ConnectionError("down")):
        allowed = check_tool_rate_limit("probe_tool", "10/minute")

    assert allowed is True
    assert limiter_module._storage_dead is True


def test_mcp_surface_stays_up_when_store_is_unreachable(client: TestClient):
    """Acceptance: an unreachable store must not close the MCP surface.

    The limit check runs ahead of the API key check, so a raising `.hit()` would
    otherwise 500 every `/mcp` request. A 401 proves the gate was passed and auth ran.
    """
    with patch.object(limiter_module._primary, "hit", side_effect=ConnectionError("down")):
        response = client.get("/mcp/sse")

    assert response.status_code == 401
    assert limiter_module._storage_dead is True
