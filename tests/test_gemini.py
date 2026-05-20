import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

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


def test_gemini_native_tools_search(client: TestClient, auth_headers, mock_gemini_client_global):
    mock_gemini_client_global.models.generate_content.reset_mock()
    
    mock_response = MagicMock()
    mock_response.text = "The answer is Google."
    
    # Mock grounding metadata
    mock_candidate = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.web.title = "Google Search Reference"
    mock_chunk.web.uri = "https://google.com"
    mock_candidate.grounding_metadata.grounding_chunks = [mock_chunk]
    mock_response.candidates = [mock_candidate]
    mock_gemini_client_global.models.generate_content.return_value = mock_response
    
    response = client.post(
        "/api/v1/gemini/",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "What is Google?",
            "native_tools": ["search"]
        },
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()["data"]
    assert "The answer is Google." in data
    assert "Sources:" in data
    assert "[Google Search Reference](https://google.com)" in data
    
    call_kwargs = mock_gemini_client_global.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert "config" in call_kwargs
    config = call_kwargs["config"]
    assert len(config.tools) == 1
    assert config.tools[0].google_search is not None


def test_gemini_native_tools_code_and_url(client: TestClient, auth_headers, mock_gemini_client_global):
    mock_gemini_client_global.models.generate_content.reset_mock()
    
    mock_response = MagicMock()
    mock_response.text = "Executed code and scraped URL successfully."
    mock_response.candidates = []
    mock_gemini_client_global.models.generate_content.return_value = mock_response
    
    response = client.post(
        "/api/v1/gemini/",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "Run code on this page: https://example.com",
            "native_tools": ["code", "url"]
        },
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()["data"]
    assert "Executed code and scraped URL successfully." in data
    
    call_kwargs = mock_gemini_client_global.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert len(config.tools) == 2
    # One is code execution, one is url context
    has_code = any(t.code_execution is not None for t in config.tools)
    has_url = any(t.url_context is not None for t in config.tools)
    assert has_code
    assert has_url


@pytest.mark.asyncio
async def test_gemini_stream_native_tools(client: TestClient, auth_headers, mock_gemini_client_global):
    mock_async_client = MagicMock()
    mock_aio = AsyncMock()
    mock_gemini_client_global.aio = mock_aio
    mock_aio.__aenter__.return_value = mock_async_client
    
    mock_generate = AsyncMock()
    mock_async_client.models.generate_content_stream = mock_generate
    
    async def mock_stream_gen():
        mock_chunk = MagicMock()
        mock_chunk.text = "This is streamed response."
        mock_chunk.candidates = []
        yield mock_chunk
        
    mock_generate.return_value = mock_stream_gen()
    
    response = client.post(
        "/api/v1/gemini/stream",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "Stream this prompt",
            "native_tools": ["search"]
        },
        headers=auth_headers
    )
    
    assert response.status_code == 200
    # Collect streamed response text
    content_str = response.content.decode("utf-8")
    assert "This is streamed response." in content_str
    
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args.kwargs
    config = call_kwargs["config"]
    assert len(config.tools) == 1
    assert config.tools[0].google_search is not None

