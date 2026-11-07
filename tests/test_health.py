"""`/api/v1/health` — T13.

The endpoint this replaces returned `{"message": "Health check passed"}` unconditionally
and the test that covered it accepted a 404, so neither could fail. These cover the
degraded and downed paths, which are the only ones worth having a health check for.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.services import health as health_module
from app.services.health import DependencyHealth


@pytest.fixture(autouse=True)
def clear_health_cache():
    """The report is cached for a few seconds; leaking it across tests hides failures."""
    health_module._cached = None
    yield
    health_module._cached = None


def _failing(message: str = "boom"):
    def _probe() -> DependencyHealth:
        raise ConnectionError(message)

    return _probe


def test_reports_every_dependency_when_healthy(client: TestClient):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "ok"
    assert set(body["data"]["dependencies"]) == {
        "database",
        "gemini",
        "pinecone",
        "rate_limit_store",
    }


def test_is_public(client: TestClient):
    """Health and root are the only routes without an API key in front of them."""
    response = client.get("/api/v1/health")

    assert response.status_code != 401


def test_downed_database_returns_503_naming_it(client: TestClient):
    """The acceptance case: a downed critical dependency must be a non-200 that says so."""
    with patch.object(health_module, "_probe_database", _failing()):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert "database" in body["error"]
    assert body["data"]["dependencies"]["database"]["status"] == "down"


def test_downed_gemini_returns_503_naming_it(client: TestClient):
    """Gemini is the product; unreachable Gemini is an outage, not a degradation."""
    with patch.object(health_module, "_probe_gemini", _failing()):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert "gemini" in response.json()["error"]


def test_downed_pinecone_is_degraded_not_an_outage(client: TestClient):
    """RAG is one router, and no §7 tool goes through it.

    Answering 503 here would take the whole service out of rotation over a dependency
    the product surface does not use.
    """
    with patch.object(health_module, "_probe_pinecone", _failing()):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["status"] == "degraded"
    assert body["data"]["dependencies"]["pinecone"]["status"] == "down"
    # Reported, but not as an outage.
    assert body["success"] is True
    assert body["error"] is None


def test_unreachable_rate_limit_store_is_degraded_not_down(client: TestClient):
    """Decision 1 made this dependency non-fatal; the health check must agree.

    Reporting 503 here would restore exactly the veto T1 removed — on Cloud Run it
    would take instances out of rotation during a rate-limit store outage.
    """
    with patch.object(
        health_module,
        "limiter_storage_status",
        return_value={"backend": "redis", "reachable": False, "degraded": True},
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["status"] == "degraded"
    assert body["data"]["dependencies"]["rate_limit_store"]["status"] == "degraded"


def test_probe_timeout_is_reported_as_down(client: TestClient):
    """One unresponsive dependency must not hold the response open."""
    import time

    def _slow() -> DependencyHealth:
        time.sleep(0.5)
        return DependencyHealth(status="ok", critical=True)

    with (
        patch.object(health_module, "_PROBE_TIMEOUT_SECONDS", 0.05),
        patch.object(health_module, "_probe_database", _slow),
    ):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert "timed out" in response.json()["data"]["dependencies"]["database"]["detail"]


def test_never_leaks_connection_details(client: TestClient):
    """This response is unauthenticated, and driver errors carry the DSN."""
    dsn = "postgresql://admin:pa55w0rd@db.internal.example.com:5432/prod"

    with patch.object(health_module, "_probe_database", _failing(f"could not connect to {dsn}")):
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert "pa55w0rd" not in response.text
    assert "db.internal.example.com" not in response.text
    # The exception type is the whole of what gets reported.
    assert response.json()["data"]["dependencies"]["database"]["detail"] == "ConnectionError"


def test_repeat_polls_are_served_from_cache(client: TestClient):
    """Unauthenticated and polled — a request loop must not fan out to every backend."""
    with patch.object(
        health_module,
        "_probe_database",
        wraps=lambda: DependencyHealth(status="ok", critical=True),
    ) as probe:
        for _ in range(5):
            client.get("/api/v1/health")

    assert probe.call_count == 1
