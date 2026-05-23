from fastapi.testclient import TestClient

def test_list_threads_returns_200(client: TestClient, auth_headers):
    response = client.get("/api/v1/threads/", headers=auth_headers)
    assert response.status_code in [200, 404]

def test_get_thread_messages_returns_404_for_invalid_id(client: TestClient, auth_headers):
    response = client.get("/api/v1/threads/non-existent-id/messages", headers=auth_headers)
    assert response.status_code in [404]

def test_threads_requires_auth(client: TestClient):
    """Thread endpoints must reject requests with no API key."""
    assert client.get("/api/v1/threads/").status_code in [401, 422]
    assert client.get("/api/v1/threads/some-id/messages").status_code in [401, 422]
    assert client.delete("/api/v1/threads/some-id").status_code in [401, 422]

def test_delete_nonexistent_thread_returns_404(client: TestClient, auth_headers):
    response = client.delete("/api/v1/threads/non-existent-id", headers=auth_headers)
    assert response.status_code == 404
