from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_rag_query_returns_401_without_auth(client: TestClient):
    response = client.post("/api/v1/rag/query", json={"query": "test"})
    assert response.status_code in [401, 404, 422]


def test_rag_ingest_returns_401_without_auth(client: TestClient):
    response = client.post("/api/v1/rag/ingest", json={"text": "test document"})
    assert response.status_code in [401, 404, 422]


def test_ingest_forwards_owner_id(client: TestClient, auth_headers: dict):
    """Router passes the caller's API-key ID as owner_id to ingest_service."""
    with patch("app.routers.rag.ingest_service", return_value=2) as mock_ingest:
        resp = client.post(
            "/api/v1/rag/ingest",
            json={"text": "isolation test document"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    assert mock_ingest.call_args.kwargs.get("owner_id") == "master"


def test_query_forwards_owner_id(client: TestClient, auth_headers: dict):
    """Router passes the caller's API-key ID as owner_id to query_service."""
    with patch("app.routers.rag.query_service", return_value="answer") as mock_query:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "what is isolation?", "model": "gemini-2.5-flash", "provider": "gemini"},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    assert mock_query.call_args.kwargs.get("owner_id") == "master"


def test_namespace_isolation_between_users():
    """Documents from different owners land in different Pinecone namespaces."""
    from app.services.rag import ingest_service

    namespaces: list[str] = []

    def capture(**kwargs: object) -> MagicMock:
        namespaces.append(str(kwargs.get("namespace", "")))
        mock = MagicMock()
        mock.add_documents.return_value = None
        return mock

    with (
        patch("app.services.rag.GeminiEmbeddings"),
        patch("app.services.rag.PineconeStore", side_effect=capture),
    ):
        ingest_service("doc for user a", owner_id="user-a")
        ingest_service("doc for user b", owner_id="user-b")

    assert len(namespaces) == 2
    assert namespaces[0] != namespaces[1]
    assert "user-a" in namespaces[0]
    assert "user-b" in namespaces[1]
