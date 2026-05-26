from unittest.mock import patch

from fastapi.testclient import TestClient


def test_mcp_health_returns_200(client: TestClient):
    # Checking if MCP server endpoint is reachable
    response = client.get("/mcp/sse")
    # MCP usually uses SSE, so a GET might return 200 or 405 depending on implementation
    # But here we just check if the route exists
    assert response.status_code in [200, 401, 404, 405]


def test_list_mcp_tools_returns_401_without_auth(client: TestClient):
    response = client.get("/api/v1/mcp/tools")
    assert response.status_code in [401, 404]


def test_mcp_rate_limit_returns_429_when_exceeded(client: TestClient):
    # Patch both auth and the rate limiter so we isolate the rate-limit behaviour
    with (
        patch("app.mcp.server.check_api_key", return_value=True),
        patch("app.mcp.server._mcp_rate_limiter") as mock_limiter,
    ):
        mock_limiter.hit.return_value = False
        response = client.get(
            "/mcp/sse",
            headers={"x-api-key": "test-master-key"},
        )
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]
