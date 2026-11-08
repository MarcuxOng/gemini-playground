"""gemini_read_url — P1.3, the second partial-overlap tool.

The overlap with a client's WebFetch is partial by design: here the page is
fetched and reasoned over by Gemini natively (url_context) in one call, and
the read pages surface as **Sources:** entries. The contracts under test:
the URL and question reach the model, the url_context tool is declared, and
the grounding chunks from the read are formatted as sources.
"""

from unittest.mock import MagicMock

import pytest

from app.services.gemini import read_url_service
from app.tools import has_tool, list_tool_names


def _web_chunk(title: str, uri: str) -> MagicMock:
    chunk = MagicMock()
    chunk.maps = None
    chunk.retrieved_context = None
    chunk.web = MagicMock(title=title, uri=uri)
    return chunk


def _mock_genai(monkeypatch: pytest.MonkeyPatch, chunks: list[MagicMock]):
    import app.services.gemini as mod
    import app.services.genai_calls as calls_mod

    candidate = MagicMock()
    candidate.grounding_metadata.grounding_chunks = chunks
    candidate.grounding_metadata.grounding_supports = []
    response = MagicMock()
    response.text = "The page explains the migration path."
    response.candidates = [candidate]
    response.prompt_feedback = None

    client = MagicMock()
    client.models.generate_content.return_value = response

    monkeypatch.setattr(mod, "client", client)
    monkeypatch.setattr(mod, "global_client", client)
    monkeypatch.setattr(calls_mod, "client", client)
    monkeypatch.setattr(calls_mod, "global_client", client)
    return client


def test_read_url_sends_url_and_question(monkeypatch: pytest.MonkeyPatch):
    client = _mock_genai(monkeypatch, [])

    read_url_service(url="https://example.com/guide", question="What changed in v2?")

    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert contents[0] == (
        "What changed in v2?\n\nRead this page and ground your answer in it: "
        "https://example.com/guide"
    )
    config = client.models.generate_content.call_args.kwargs["config"]
    assert any(t.url_context is not None for t in config.tools), config.tools


def test_read_url_appends_the_read_page_as_source(monkeypatch: pytest.MonkeyPatch):
    _mock_genai(
        monkeypatch,
        [_web_chunk("Migration guide", "https://example.com/guide")],
    )

    answer = read_url_service(url="https://example.com/guide", question="What changed in v2?")

    assert answer.startswith("The page explains the migration path.")
    assert "**Sources:**" in answer
    assert "[Migration guide](https://example.com/guide)" in answer


def test_read_url_without_chunks_returns_plain_answer(monkeypatch: pytest.MonkeyPatch):
    _mock_genai(monkeypatch, [])

    answer = read_url_service(url="https://example.com/guide", question="q?")

    assert answer == "The page explains the migration path."
    assert "Sources" not in answer


def test_gemini_read_url_is_registered():
    """The tool must be in the registry so _register_all_tools() auto-exposes it on /mcp/sse."""
    assert "gemini_read_url" in list_tool_names()
    assert has_tool("gemini_read_url")
