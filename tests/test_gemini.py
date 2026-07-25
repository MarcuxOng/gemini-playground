from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

# --- Model name validation ---

@pytest.mark.parametrize("bad_model", [
    "../../etc/passwd",
    "gpt-4",
    "claude-3",
    "openai/gpt-4",
    "",
    "GEMINI-flash",
])
def test_gemini_rejects_invalid_model_names(client: TestClient, auth_headers, bad_model: str):
    response = client.post(
        "/api/v1/gemini/",
        json={"model": bad_model, "prompt": "hello"},
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.parametrize("good_model", [
    "gemini-2.5-flash",
    "gemini-1.5-pro",
    "text-embedding-004",
])
def test_gemini_accepts_valid_model_names(client: TestClient, auth_headers, mock_gemini_client_global, good_model: str):
    response = client.post(
        "/api/v1/gemini/",
        json={"model": good_model, "prompt": "hello"},
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.parametrize("bad_field,bad_value", [
    ("temperature", 2.1),
    ("temperature", -0.1),
    ("top_p", 1.1),
    ("top_p", -0.1),
    ("top_k", 0),
    ("thinking_budget", -2),
])
def test_gemini_rejects_out_of_range_sampling_params(client: TestClient, auth_headers, bad_field: str, bad_value: float):
    response = client.post(
        "/api/v1/gemini/",
        json={"model": "gemini-2.5-flash", "prompt": "hello", bad_field: bad_value},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_gemini_structured_rejects_invalid_model(client: TestClient, auth_headers):
    response = client.post(
        "/api/v1/gemini/structured",
        json={"model": "gpt-4", "prompt": "hello", "response_schema": {"type": "object"}},
        headers=auth_headers,
    )
    assert response.status_code == 422


# --- Existing tests ---

def _fake_model(name: str, actions: list[str]) -> MagicMock:
    m = MagicMock()
    m.name = f"models/{name}"
    m.supported_actions = actions
    return m


def test_list_gemini_models_merges_regional_and_global(
    client: TestClient, auth_headers, mock_gemini_client_global: MagicMock
):
    """Models unique to either endpoint are both surfaced, with capability metadata."""
    import app.services.gemini as gemini_module

    mock_gemini_client_global.models.list.return_value = [
        _fake_model("gemini-2.5-flash", ["generateContent", "countTokens"]),
        _fake_model("gemini-live-2.5-flash-native-audio", ["bidiGenerateContent"]),  # regional-only
    ]
    mock_global_client = MagicMock()
    mock_global_client.models.list.return_value = [
        _fake_model("gemini-2.5-flash", ["generateContent", "countTokens", "createCachedContent"]),
        _fake_model("gemini-3.5-flash", ["generateContent", "countTokens"]),  # global-only
    ]

    try:
        with patch.object(gemini_module, "global_client", mock_global_client):
            response = client.get("/api/v1/gemini/models", headers=auth_headers)
    finally:
        mock_gemini_client_global.models.list.reset_mock(return_value=True, side_effect=True)

    assert response.status_code == 200
    models = {m["name"]: set(m["supported_actions"]) for m in response.json()["data"]}
    assert set(models) == {
        "gemini-2.5-flash",
        "gemini-live-2.5-flash-native-audio",
        "gemini-3.5-flash",
    }
    # Actions for a model listed on both endpoints are unioned.
    assert models["gemini-2.5-flash"] == {"generateContent", "countTokens", "createCachedContent"}


def test_list_gemini_models_tolerates_one_endpoint_failing(
    client: TestClient, auth_headers, mock_gemini_client_global: MagicMock
):
    """If the global endpoint errors, regional models still come back rather than 500ing."""
    import app.services.gemini as gemini_module

    mock_gemini_client_global.models.list.return_value = [
        _fake_model("gemini-2.5-flash", ["generateContent"]),
    ]
    mock_global_client = MagicMock()
    mock_global_client.models.list.side_effect = errors.ClientError(
        503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "global endpoint down"}}
    )

    try:
        with patch.object(gemini_module, "global_client", mock_global_client):
            response = client.get("/api/v1/gemini/models", headers=auth_headers)
    finally:
        mock_gemini_client_global.models.list.reset_mock(return_value=True, side_effect=True)

    assert response.status_code == 200
    assert [m["name"] for m in response.json()["data"]] == ["gemini-2.5-flash"]

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
    mock_response.prompt_feedback = None
    
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
    mock_response.prompt_feedback = None
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


def test_gemini_stop_sequences_and_system_instruction(client: TestClient, auth_headers, mock_gemini_client_global):
    mock_gemini_client_global.models.generate_content.reset_mock()

    mock_response = MagicMock()
    mock_response.text = "Response respecting the system instruction."
    mock_response.candidates = []
    mock_response.prompt_feedback = None
    mock_gemini_client_global.models.generate_content.return_value = mock_response

    response = client.post(
        "/api/v1/gemini/",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "Say hello",
            "stop_sequences": ["STOP"],
            "system_instruction": "Always respond in French.",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    call_kwargs = mock_gemini_client_global.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.stop_sequences == ["STOP"]
    assert config.system_instruction == "Always respond in French."


def test_gemini_sampling_params_use_langchain_path(client: TestClient, auth_headers, mock_gemini_client_global):
    """With no attachments/native_tools/cache_id/stop_sequences/system_instruction present,
    sampling params should route through build_llm() (LangChain path), not the raw client."""
    import app.services.gemini as gemini_module

    response = client.post(
        "/api/v1/gemini/",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "Say hello",
            "temperature": 0.9,
            "top_p": 0.8,
            "top_k": 20,
            "seed": 42,
            "thinking_budget": 1024,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    call_kwargs = gemini_module.build_llm.call_args.kwargs
    assert call_kwargs["temperature"] == 0.9
    assert call_kwargs["top_p"] == 0.8
    assert call_kwargs["top_k"] == 20
    assert call_kwargs["seed"] == 42
    assert call_kwargs["thinking_budget"] == 1024


def test_gemini_sampling_params_raw_client_path(client: TestClient, auth_headers, mock_gemini_client_global):
    """Combined with system_instruction (which forces the raw client path), sampling params
    must still reach GenerateContentConfig rather than being dropped."""
    mock_gemini_client_global.models.generate_content.reset_mock()

    mock_response = MagicMock()
    mock_response.text = "Response with custom sampling."
    mock_response.candidates = []
    mock_response.prompt_feedback = None
    mock_gemini_client_global.models.generate_content.return_value = mock_response

    response = client.post(
        "/api/v1/gemini/",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "Say hello",
            "system_instruction": "Be terse.",
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 10,
            "seed": 7,
            "thinking_budget": 0,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    call_kwargs = mock_gemini_client_global.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.temperature == 0.4
    assert config.top_p == 0.9
    assert config.top_k == 10
    assert config.seed == 7
    assert config.thinking_config.thinking_budget == 0


@pytest.mark.asyncio
async def test_gemini_stream_native_tools(client: TestClient, auth_headers, mock_gemini_client_global):
    mock_aio = MagicMock()
    mock_gemini_client_global.aio = mock_aio

    mock_generate = AsyncMock()
    mock_aio.models.generate_content_stream = mock_generate
    
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


@pytest.mark.asyncio
async def test_gemini_stream_sampling_params(client: TestClient, auth_headers, mock_gemini_client_global):
    """Streaming always uses the raw client, so sampling params should reach
    GenerateContentConfig even with no other raw-client trigger present."""
    mock_aio = MagicMock()
    mock_gemini_client_global.aio = mock_aio

    mock_generate = AsyncMock()
    mock_aio.models.generate_content_stream = mock_generate

    async def mock_stream_gen():
        mock_chunk = MagicMock()
        mock_chunk.text = "Streamed with custom sampling."
        mock_chunk.candidates = []
        yield mock_chunk

    mock_generate.return_value = mock_stream_gen()

    response = client.post(
        "/api/v1/gemini/stream",
        json={
            "model": "gemini-2.5-flash",
            "prompt": "Stream this",
            "temperature": 1.2,
            "top_p": 0.75,
            "top_k": 30,
            "seed": 99,
            "thinking_budget": -1,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    content_str = response.content.decode("utf-8")
    assert "Streamed with custom sampling." in content_str

    call_kwargs = mock_generate.call_args.kwargs
    config = call_kwargs["config"]
    assert config.temperature == 1.2
    assert config.top_p == 0.75
    assert config.top_k == 30
    assert config.seed == 99
    assert config.thinking_config.thinking_budget == -1


# --- Regional/global fallback (models only served from one Vertex endpoint) ---

def test_gemini_falls_back_to_global_on_regional_404(client: TestClient, auth_headers, mock_gemini_client_global):
    import app.services.gemini as gemini_module

    mock_gemini_client_global.models.generate_content.reset_mock(side_effect=True)
    mock_gemini_client_global.models.generate_content.side_effect = errors.ClientError(
        404,
        {"error": {"code": 404, "status": "NOT_FOUND", "message": "Publisher model not found in region"}},
    )

    mock_global_response = MagicMock()
    mock_global_response.text = "Response from the global endpoint."
    mock_global_response.candidates = []
    mock_global_response.prompt_feedback = None
    mock_global_client = MagicMock()
    mock_global_client.models.generate_content.return_value = mock_global_response

    try:
        with patch.object(gemini_module, "global_client", mock_global_client):
            response = client.post(
                "/api/v1/gemini/",
                json={
                    "model": "gemini-3.5-flash",
                    "prompt": "Say hello",
                    "system_instruction": "Be terse.",  # forces the raw-client path
                },
                headers=auth_headers,
            )
    finally:
        # This mock is a session-scoped shared fixture (patched into other services too) —
        # clear the side_effect so it doesn't leak into unrelated tests.
        mock_gemini_client_global.models.generate_content.side_effect = None

    assert response.status_code == 200
    assert "Response from the global endpoint." in response.json()["data"]
    mock_global_client.models.generate_content.assert_called_once()
    assert mock_global_client.models.generate_content.call_args.kwargs["model"] == "gemini-3.5-flash"


def test_gemini_raw_client_non_404_error_does_not_fall_back(client: TestClient, auth_headers, mock_gemini_client_global):
    import app.services.gemini as gemini_module

    mock_gemini_client_global.models.generate_content.reset_mock(side_effect=True)
    mock_gemini_client_global.models.generate_content.side_effect = errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}},
    )
    mock_global_client = MagicMock()

    try:
        with patch.object(gemini_module, "global_client", mock_global_client):
            # Non-404 errors propagate unhandled (matching production behavior), not a clean 500 response.
            with pytest.raises(errors.ClientError) as exc_info:
                client.post(
                    "/api/v1/gemini/",
                    json={
                        "model": "gemini-2.5-flash",
                        "prompt": "Say hello",
                        "system_instruction": "Be terse.",
                    },
                    headers=auth_headers,
                )
            assert exc_info.value.code == 429
    finally:
        mock_gemini_client_global.models.generate_content.side_effect = None

    mock_global_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_gemini_stream_falls_back_to_global_on_regional_404(client: TestClient, auth_headers, mock_gemini_client_global):
    import app.services.gemini as gemini_module

    mock_aio = MagicMock()
    mock_gemini_client_global.aio = mock_aio
    mock_generate = AsyncMock()
    mock_aio.models.generate_content_stream = mock_generate
    mock_generate.side_effect = errors.ClientError(
        404,
        {"error": {"code": 404, "status": "NOT_FOUND", "message": "Publisher model not found in region"}},
    )

    async def mock_global_stream_gen():
        mock_chunk = MagicMock()
        mock_chunk.text = "Streamed from global."
        mock_chunk.candidates = []
        yield mock_chunk

    mock_global_client = MagicMock()
    mock_global_generate = AsyncMock()
    mock_global_client.aio.models.generate_content_stream = mock_global_generate
    mock_global_generate.return_value = mock_global_stream_gen()

    with patch.object(gemini_module, "global_client", mock_global_client):
        response = client.post(
            "/api/v1/gemini/stream",
            json={"model": "gemini-3.5-flash", "prompt": "Stream this"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "Streamed from global." in response.content.decode("utf-8")
    mock_global_generate.assert_called_once()

