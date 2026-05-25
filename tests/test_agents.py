import pytest
from fastapi.testclient import TestClient


def test_list_agents_returns_200(client: TestClient, auth_headers):
    response = client.get("/api/v1/agents/list", headers=auth_headers)
    assert response.status_code == 200


def test_run_agent_returns_401_without_auth(client: TestClient):
    response = client.post("/api/v1/agents/run", json={"prompt": "hello", "preset": "coder"})
    assert response.status_code in [401, 422]


@pytest.mark.parametrize("bad_model", ["gpt-4", "../../etc/passwd", "claude-3"])
def test_run_agent_rejects_invalid_model(client: TestClient, auth_headers, bad_model: str):
    response = client.post(
        "/api/v1/agents/run",
        json={"model": bad_model, "prompt": "hello", "preset": "research"},
        headers=auth_headers,
    )
    assert response.status_code == 422
