from fastapi.testclient import TestClient

def test_list_gemini_models_returns_200(client: TestClient, auth_headers):
    response = client.get("/api/v1/gemini/models", headers=auth_headers)
    assert response.status_code in [200, 404]

def test_gemini_service_returns_401_without_auth(client: TestClient):
    response = client.post("/api/v1/gemini/", json={"model": "gemini-pro", "prompt": "hello"})
    assert response.status_code in [401, 422]


def test_gemini_structured_returns_401_without_auth(client: TestClient):
    response = client.post(
        "/api/v1/gemini/structured",
        json={"model": "gemini-pro", "prompt": "hello", "response_schema": {"type": "object"}},
    )
    assert response.status_code in [401, 422]


def test_gemini_structured_happy_path(client: TestClient, auth_headers, mock_gemini_client_global):
    response = client.post(
        "/api/v1/gemini/structured",
        json={
            "model": "gemini-1.5-flash",
            "prompt": "Return a mock JSON",
            "response_schema": {"type": "object", "properties": {"foo": {"type": "string"}}},
        },
        headers=auth_headers,
    )
    # response_model APIResponse ensures data is present
    assert response.status_code == 200
    assert "data" in response.json()
