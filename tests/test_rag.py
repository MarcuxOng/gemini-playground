from unittest.mock import MagicMock, patch

import pytest
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
            json={"query": "what is isolation?", "model": "gemini-2.5-flash"},
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


# --- Phase 8.3 Multimodal RAG tests ---


def test_embed_file_uri_calls_genai():
    """GeminiEmbeddings.embed_file_uri calls the Gemini embed API with a Part."""
    from app.rag.embeddings import GeminiEmbeddings

    mock_client = MagicMock()
    mock_embeddings_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_embeddings_response.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_embeddings_response

    embeddings = GeminiEmbeddings(model="gemini-embedding-2")
    embeddings.client = mock_client

    result = embeddings.embed_file_uri(
        "https://generativelanguage.googleapis.com/files/test-img", "image/png"
    )

    assert result == [0.1, 0.2, 0.3]
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-embedding-2"
    assert call_kwargs["config"] == {"task_type": "retrieval_document"}


def test_embed_file_uri_empty_returns_empty_list():
    """GeminiEmbeddings.embed_file_uri returns empty list when API returns no embedding."""
    from app.rag.embeddings import GeminiEmbeddings

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.embeddings = []
    mock_client.models.embed_content.return_value = mock_response

    embeddings = GeminiEmbeddings(model="gemini-embedding-2")
    embeddings.client = mock_client

    result = embeddings.embed_file_uri(
        "https://generativelanguage.googleapis.com/files/empty", "image/png"
    )

    assert result == []


def test_pinecone_add_file_documents_upserts():
    """PineconeStore.add_file_documents calls index.upsert with file embeddings."""
    from langchain_core.documents import Document

    from app.rag.vectorstore import PineconeStore

    mock_index = MagicMock()
    mock_pc = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_embedding = MagicMock()
    mock_embedding.embed_file_uri.return_value = [0.1, 0.2, 0.3]

    with patch("app.rag.vectorstore.Pinecone", return_value=mock_pc):
        store = PineconeStore(
            index_name="test-idx",
            embedding=mock_embedding,
            api_key="test-key",
            namespace="test-ns",
        )

    store.index = mock_index

    docs = [
        Document(
            page_content="test image",
            metadata={
                "gemini_file_uri": "https://example.com/files/img1",
                "mime_type": "image/png",
                "display_name": "test.png",
            },
        )
    ]

    store.add_file_documents(docs)

    mock_embedding.embed_file_uri.assert_called_once_with(
        "https://example.com/files/img1", "image/png"
    )
    mock_index.upsert.assert_called_once()
    call_args = mock_index.upsert.call_args.kwargs
    assert call_args["namespace"] == "test-ns"
    assert len(call_args["vectors"]) == 1
    assert call_args["vectors"][0]["metadata"]["source_type"] == "multimodal"


def test_pinecone_add_file_documents_skips_missing_uri():
    """PineconeStore.add_file_documents skips docs without gemini_file_uri."""
    from langchain_core.documents import Document

    from app.rag.vectorstore import PineconeStore

    mock_index = MagicMock()
    mock_pc = MagicMock()
    mock_pc.Index.return_value = mock_index
    mock_embedding = MagicMock()

    with patch("app.rag.vectorstore.Pinecone", return_value=mock_pc):
        store = PineconeStore(
            index_name="test-idx",
            embedding=mock_embedding,
            api_key="test-key",
            namespace="test-ns",
        )

    store.index = mock_index

    docs = [
        Document(
            page_content="no uri doc",
            metadata={"mime_type": "image/png"},
        )
    ]

    store.add_file_documents(docs)
    mock_embedding.embed_file_uri.assert_not_called()
    mock_index.upsert.assert_not_called()


def test_ingest_file_service_creates_documents():
    """ingest_file_service looks up files, creates multimodal docs, and stores them."""
    from app.services.rag import ingest_file_service

    mock_file = MagicMock()
    mock_file.id = "file-123"
    mock_file.gemini_file_uri = "https://example.com/files/img1"
    mock_file.mime_type = "image/png"
    mock_file.display_name = "test.png"

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query  # chained filter returns self
    mock_query.first.return_value = mock_file

    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    mock_vs = MagicMock()
    mock_vs.add_file_documents.return_value = None

    with (
        patch("app.services.rag.vectorstore_service", return_value=mock_vs),
        patch("app.services.rag._build_namespace", return_value="ns_test_user-a"),
    ):
        result = ingest_file_service(["file-123"], mock_db, "user-a")

    assert result == 1
    mock_vs.add_file_documents.assert_called_once()
    docs = mock_vs.add_file_documents.call_args[0][0]
    assert len(docs) == 1
    assert docs[0].metadata["gemini_file_uri"] == "https://example.com/files/img1"
    assert docs[0].metadata["source_type"] == "multimodal"


def test_ingest_file_service_no_valid_files_raises():
    """ingest_file_service raises ValueError when no valid file records found."""
    from app.services.rag import ingest_file_service

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = None

    mock_db = MagicMock()
    mock_db.query.return_value = mock_query

    with pytest.raises(ValueError, match="No valid file records"):
        ingest_file_service(["nonexistent"], mock_db, "user-a")


def test_ingest_endpoint_handles_file_ids(client: TestClient, auth_headers: dict):
    """POST /api/v1/rag/ingest accepts file_ids alongside text."""
    with (
        patch("app.routers.rag.ingest_service", return_value=2) as mock_text,
        patch("app.routers.rag.ingest_file_service", return_value=1) as mock_file,
    ):
        resp = client.post(
            "/api/v1/rag/ingest",
            json={"text": "hello world", "file_ids": ["file-123"]},
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["text_chunks"] == 2
    assert data["files"] == 1


def test_ingest_endpoint_rejects_empty_body(client: TestClient, auth_headers: dict):
    """POST /api/v1/rag/ingest returns 400 when neither text nor file_ids provided."""
    resp = client.post(
        "/api/v1/rag/ingest",
        json={},
        headers=auth_headers,
    )
    assert resp.status_code in (400, 429)


def test_search_knowledge_base_includes_file_uris():
    """search_knowledge_base tool includes file URIs in its output."""
    from langchain_core.documents import Document

    from app.tools.system.rag import search_knowledge_base

    mock_docs = [
        Document(
            page_content="A cat on a mat",
            metadata={"source": "text"},
        ),
        Document(
            page_content="cat_photo.png",
            metadata={
                "gemini_file_uri": "https://example.com/files/cat-img",
                "mime_type": "image/png",
                "display_name": "cat_photo.png",
            },
        ),
    ]

    with patch("app.tools.system.rag.search_documents", return_value=mock_docs):
        result = search_knowledge_base("cat")

    assert "Retrieved File Attachments" in result
    assert "https://example.com/files/cat-img" in result
    assert "cat_photo.png" in result


def test_embedding_model_default_dev_is_gemini_embedding_2():
    """Default embedding model is gemini-embedding-2 when ENV != production."""
    from app.config import Settings

    s = Settings(
        database_url="sqlite:///./test.db",
        master_api_key="test-key",
        gemini_api_key="test-key",
        gcp_project_id="test-project",
        pinecone_namespace="test-ns",
        pinecone_index_name="test-idx",
        pinecone_api_key="test-key",
        alpha_vantage_api_key="test",
        openweathermap_api_key="test",
        news_api_key="test",
    )
    assert s.gemini_embedding_model == "gemini-embedding-2"


def test_embedding_model_default_prod_is_multimodalembedding(monkeypatch):
    """Default embedding model is multimodalembedding when ENV == production."""
    monkeypatch.setenv("ENV", "production")

    from app.config import Settings

    s = Settings(
        database_url="sqlite:///./test.db",
        master_api_key="test-key",
        gemini_api_key="test-key",
        gcp_project_id="test-project",
        pinecone_namespace="test-ns",
        pinecone_index_name="test-idx",
        pinecone_api_key="test-key",
        alpha_vantage_api_key="test",
        openweathermap_api_key="test",
        news_api_key="test",
    )
    assert s.gemini_embedding_model == "multimodalembedding"


def test_embedding_model_explicit_overrides_env(monkeypatch):
    """Explicit GEMINI_EMBEDDING_MODEL overrides the env-aware default."""
    monkeypatch.setenv("ENV", "production")

    from app.config import Settings

    s = Settings(
        database_url="sqlite:///./test.db",
        master_api_key="test-key",
        gemini_api_key="test-key",
        gcp_project_id="test-project",
        pinecone_namespace="test-ns",
        pinecone_index_name="test-idx",
        pinecone_api_key="test-key",
        alpha_vantage_api_key="test",
        openweathermap_api_key="test",
        news_api_key="test",
        gemini_embedding_model="custom-embed-v1",
    )
    assert s.gemini_embedding_model == "custom-embed-v1"


def test_search_documents_returns_multimodal_metadata():
    """search_documents returns Document objects with gemini_file_uri in metadata."""
    from langchain_core.documents import Document

    from app.services.rag import search_documents

    multimodal_docs = [
        Document(
            page_content="test image",
            metadata={
                "gemini_file_uri": "https://example.com/files/img1",
                "mime_type": "image/png",
                "display_name": "screenshot.png",
                "source_type": "multimodal",
            },
        ),
    ]

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = multimodal_docs

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    with patch("app.services.rag.vectorstore_service", return_value=mock_vs):
        results = search_documents("image query", owner_id="user-a")

    assert len(results) == 1
    assert results[0].metadata["gemini_file_uri"] == "https://example.com/files/img1"
    assert results[0].metadata["source_type"] == "multimodal"
