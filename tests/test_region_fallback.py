"""The regional -> global Vertex fallback, both call paths in one place.

Some models serve only from the `global` endpoint — `gemini-embedding-2` works there
and 404s on `us-central1` — so every call path has to retry globally on a regional 404
and, just as importantly, *not* retry on anything else. This is the behaviour T4 tracks
as the "region 404 tax".

The guarantee used to be verified from three files at once: `test_gemini.py` covered the
raw `genai` path, `test_rag.py` covered it again through `query_service`, and
`test_llm_region_fallback.py` covered the LangChain path (finding F9). Nothing said
whether the fallback was covered end to end. It is one guarantee over one module —
`app/services/genai_calls.py` owns the raw side since F3 — so it belongs in one file.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import NotFound
from google.genai import errors
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableWithFallbacks

import app.services.genai_calls as genai_calls_module
from app.config import settings
from app.services.llm import build_llm_with_region_fallback
from app.services.rag import query_service

REGIONAL_404 = {
    "error": {"code": 404, "status": "NOT_FOUND", "message": "Publisher model not found in region"}
}
QUOTA_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}


@pytest.fixture
def regional_client_fails(mock_gemini_client_global):
    """Make the regional client raise, then undo it.

    `mock_gemini_client_global` is session-scoped and patched into several services, so a
    `side_effect` left behind leaks into every later test that touches it. Three copies of
    this try/finally lived in three files; one fixture makes the reset unforgettable.
    """

    def _arrange(status_code: int, payload: dict) -> None:
        mock_gemini_client_global.models.generate_content.reset_mock(side_effect=True)
        mock_gemini_client_global.models.generate_content.side_effect = errors.ClientError(
            status_code, payload
        )

    yield _arrange
    mock_gemini_client_global.models.generate_content.side_effect = None


def _global_client_returning(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.candidates = []
    response.prompt_feedback = None
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


# --- Raw `genai` path — app/services/genai_calls.py --------------------------------


def test_generate_falls_back_to_global_on_regional_404(
    client: TestClient, auth_headers, regional_client_fails
):
    regional_client_fails(404, REGIONAL_404)
    global_client = _global_client_returning("Response from the global endpoint.")

    with patch.object(genai_calls_module, "global_client", global_client):
        response = client.post(
            "/api/v1/gemini/",
            json={
                "model": "gemini-3.5-flash",
                "prompt": "Say hello",
                "system_instruction": "Be terse.",  # forces the raw-client path
            },
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "Response from the global endpoint." in response.json()["data"]
    global_client.models.generate_content.assert_called_once()
    assert global_client.models.generate_content.call_args.kwargs["model"] == "gemini-3.5-flash"


def test_generate_does_not_fall_back_on_a_non_404(
    client: TestClient, auth_headers, regional_client_fails
):
    """A quota error is not a region problem — retrying globally would just burn it twice."""
    regional_client_fails(429, QUOTA_429)
    global_client = MagicMock()

    with patch.object(genai_calls_module, "global_client", global_client):
        # Non-404s propagate unhandled, matching production behaviour.
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
    global_client.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_stream_falls_back_to_global_on_regional_404(
    client: TestClient, auth_headers, mock_gemini_client_global
):
    """Streaming goes through `aio`, a different attribute than the sync path."""
    mock_aio = MagicMock()
    mock_gemini_client_global.aio = mock_aio
    mock_aio.models.generate_content_stream = AsyncMock(
        side_effect=errors.ClientError(404, REGIONAL_404)
    )

    async def global_stream():
        chunk = MagicMock()
        chunk.text = "Streamed from global."
        chunk.candidates = []
        yield chunk

    global_client = MagicMock()
    global_generate = AsyncMock(return_value=global_stream())
    global_client.aio.models.generate_content_stream = global_generate

    with patch.object(genai_calls_module, "global_client", global_client):
        response = client.post(
            "/api/v1/gemini/stream",
            json={"model": "gemini-3.5-flash", "prompt": "Stream this"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "Streamed from global." in response.content.decode("utf-8")
    global_generate.assert_called_once()


def test_rag_multimodal_query_falls_back_to_global_on_regional_404(regional_client_fails):
    """The multimodal RAG branch reaches the raw client directly, not through a router."""
    regional_client_fails(404, REGIONAL_404)
    global_client = _global_client_returning("Answer from the global endpoint.")

    file_docs = [
        Document(
            page_content="",
            metadata={
                "gemini_file_uri": "https://example.com/files/img1",
                "mime_type": "image/png",
                "display_name": "screenshot.png",
            },
        ),
    ]

    with (
        patch("app.services.rag.search_documents", return_value=file_docs),
        patch.object(genai_calls_module, "global_client", global_client),
    ):
        result = query_service("what is in the image?", "gemini-3.5-flash", owner_id="user-a")

    assert result == "Answer from the global endpoint."
    global_client.models.generate_content.assert_called_once()


# --- LangChain path — app/services/llm.py (finding F9) ------------------------------


class FakeModel(Runnable):  # type: ignore[type-arg]
    """Minimal real Runnable — RunnableWithFallbacks pydantic-validates its members."""

    def __init__(
        self, location: str | None, result: str = "", error: BaseException | None = None
    ) -> None:
        self.location = location
        self.result = result
        self.error = error
        self.calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_langchain_composes_regional_primary_with_global_fallback():
    regional = FakeModel(settings.gcp_region)
    global_model = FakeModel("global")

    with patch("app.services.llm.build_llm", side_effect=[regional, global_model]) as mock_build:
        composed = build_llm_with_region_fallback("gemini-2.5-flash")

    assert isinstance(composed, RunnableWithFallbacks)
    # The second build must explicitly ask for the global endpoint.
    assert mock_build.call_args_list[1].kwargs["location"] == "global"


def test_langchain_falls_back_to_global_when_regional_raises_not_found():
    regional = FakeModel(settings.gcp_region, error=NotFound("model not found in region"))
    global_model = FakeModel("global", result="answer from global")

    with patch("app.services.llm.build_llm", side_effect=[regional, global_model]):
        composed = build_llm_with_region_fallback("gemini-embedding-2")

    assert composed.invoke("hi") == "answer from global"
    assert regional.calls == 1
    assert global_model.calls == 1


def test_langchain_does_not_fall_back_on_non_404_errors():
    regional = FakeModel(settings.gcp_region, error=RuntimeError("quota exhausted"))
    global_model = FakeModel("global", result="should not be reached")

    with patch("app.services.llm.build_llm", side_effect=[regional, global_model]):
        composed = build_llm_with_region_fallback("gemini-2.5-flash")

    with pytest.raises(RuntimeError, match="quota exhausted"):
        composed.invoke("hi")

    assert global_model.calls == 0


def test_ai_studio_path_returns_bare_model():
    """The dev path has no regions, so there is nothing to fall back to."""
    dev_model = FakeModel(None)

    with patch("app.services.llm.build_llm", side_effect=[dev_model]) as mock_build:
        composed = build_llm_with_region_fallback("gemini-2.5-flash")

    assert composed is dev_model
    assert mock_build.call_count == 1
