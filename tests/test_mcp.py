"""The MCP surface's gate — `MCPAuthMiddleware`.

Two of the three tests here used to accept `404`. One asked for `/api/v1/mcp/tools`,
which has never existed (the router is `/api/v1/mcp-servers`), and the other was named
`test_mcp_health_returns_200` while accepting any of 200, 401, 404 or 405.

Only the rejection paths are exercised on `/mcp/sse`: a request that passes the gate
opens an SSE stream, which a TestClient `get` would sit on.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_mcp_requires_an_api_key(client: TestClient):
    response = client.get("/mcp/sse")

    assert response.status_code == 401
    assert response.json()["error"] == "Unauthorized: Missing API Key"


def test_mcp_rejects_an_unknown_api_key(client: TestClient):
    """A present-but-invalid key is a different branch from a missing one."""
    response = client.get("/mcp/sse", headers={"x-api-key": "not-a-real-key"})

    assert response.status_code == 401
    assert response.json()["error"] == "Unauthorized: Invalid API Key"


def test_mcp_gate_only_covers_the_mcp_path(client: TestClient):
    """The middleware is mounted app-wide and must not gate the REST surface."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_mcp_rate_limit_returns_429_when_exceeded(client: TestClient):
    # Patch both auth and the rate limiter so we isolate the rate-limit behaviour
    with (
        patch("app.mcp.server.check_api_key", return_value=True),
        patch("app.mcp.server.limiter_hit", return_value=False),
    ):
        response = client.get(
            "/mcp/sse",
            headers={"x-api-key": "test-master-key"},
        )
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]


def test_mcp_servers_router_requires_auth(client: TestClient):
    """The admin router for external MCP servers — the route the old test meant to hit."""
    assert client.get("/api/v1/mcp-servers/").status_code == 401
    assert client.post("/api/v1/mcp-servers/", json={}).status_code == 401
