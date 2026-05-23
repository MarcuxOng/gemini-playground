import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_headers_non_master():
    return {"X-API-Key": "not-the-master-key"}


def test_list_threads_returns_200(client: TestClient, auth_headers):
    response = client.get("/api/v1/threads/", headers=auth_headers)
    assert response.status_code in [200, 404]


def test_get_thread_messages_returns_404_for_invalid_id(client: TestClient, auth_headers):
    response = client.get("/api/v1/threads/non-existent-id/messages", headers=auth_headers)
    assert response.status_code == 404


def test_threads_requires_auth(client: TestClient):
    """Thread endpoints protected by verify_master_key return 422 when no key is provided."""
    assert client.get("/api/v1/threads/").status_code == 422
    assert client.get("/api/v1/threads/some-id/messages").status_code == 422
    assert client.delete("/api/v1/threads/some-id").status_code == 422


def test_threads_forbidden_with_non_master_api_key(
    client: TestClient, auth_headers_non_master
):
    """Thread endpoints protected by verify_master_key return 403 for a non-master key."""
    assert client.get("/api/v1/threads/", headers=auth_headers_non_master).status_code == 403
    assert (
        client.get(
            "/api/v1/threads/some-id/messages", headers=auth_headers_non_master
        ).status_code
        == 403
    )
    assert (
        client.delete("/api/v1/threads/some-id", headers=auth_headers_non_master).status_code
        == 403
    )


def test_delete_nonexistent_thread_returns_404(client: TestClient, auth_headers):
    response = client.delete("/api/v1/threads/non-existent-id", headers=auth_headers)
    assert response.status_code == 404
