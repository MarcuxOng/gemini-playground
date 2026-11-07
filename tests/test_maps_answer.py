"""gemini_maps_answer — P1.2, the second no-overlap Option B tool.

The service is tested through its real code path with only the genai client
mocked (no mocking the function under test). Maps grounding chunks are a
distinct response shape from web chunks — `chunk.maps` carries place_id,
title and uri — and the contract under test is that they are surfaced as
**Places:** citations rather than silently dropped, which is what happened
before this tool existed.
"""

from unittest.mock import MagicMock

import pytest
from google.genai import types

from app.services.gemini import maps_answer_service
from app.tools import has_tool, list_tool_names


def _maps_chunk(title: str, place_id: str, uri: str) -> MagicMock:
    """Build a grounding chunk the way the SDK returns a Maps result.

    types.GroundingChunk is pydantic, but the service branches on
    `chunk.web` / `chunk.maps` attribute presence, so a MagicMock with the
    right attribute tree exercises the same code path.
    """
    chunk = MagicMock()
    chunk.web = None
    chunk.maps = types.GroundingChunkMaps(title=title, place_id=place_id, uri=uri)
    return chunk


def _mock_genai(monkeypatch: pytest.MonkeyPatch, chunks: list[MagicMock]):
    import app.services.gemini as mod
    import app.services.genai_calls as calls_mod

    candidate = MagicMock()
    candidate.grounding_metadata.grounding_chunks = chunks
    response = MagicMock()
    response.text = "Try Kai Lane Coffee — quiet, good pour-overs."
    response.candidates = [candidate]
    response.prompt_feedback = None

    client = MagicMock()
    client.models.generate_content.return_value = response

    monkeypatch.setattr(mod, "client", client)
    monkeypatch.setattr(mod, "global_client", client)
    monkeypatch.setattr(calls_mod, "client", client)
    monkeypatch.setattr(calls_mod, "global_client", client)
    return client


def test_maps_answer_appends_place_citations(monkeypatch: pytest.MonkeyPatch):
    client = _mock_genai(
        monkeypatch,
        [
            _maps_chunk(
                "Kai Lane Coffee",
                "place-123",
                "https://maps.google.com/?cid=123",
            ),
        ],
    )

    answer = maps_answer_service("quiet cafe near Raffles Place")

    assert answer.startswith("Try Kai Lane Coffee — quiet, good pour-overs.")
    assert "**Places:**" in answer
    assert "[Kai Lane Coffee](https://maps.google.com/?cid=123)" in answer
    assert "(place_id: place-123)" in answer

    # The request must actually carry the google_maps native tool.
    config = client.models.generate_content.call_args.kwargs["config"]
    tool_types = [t.google_maps is not None for t in config.tools]
    assert any(tool_types), f"expected a google_maps tool, got {config.tools}"


def test_maps_answer_dedupes_places(monkeypatch: pytest.MonkeyPatch):
    _mock_genai(
        monkeypatch,
        [
            _maps_chunk("Kai Lane Coffee", "place-123", "https://maps.google.com/?cid=123"),
            _maps_chunk("Kai Lane Coffee", "place-123", "https://maps.google.com/?cid=123"),
        ],
    )

    answer = maps_answer_service("quiet cafe")

    assert answer.count("[Kai Lane Coffee](https://maps.google.com/?cid=123)") == 1


def test_maps_answer_web_chunks_still_work_alongside(monkeypatch: pytest.MonkeyPatch):
    """A grounded answer can cite both web and maps sources in one response."""
    web_chunk = MagicMock()
    web_chunk.maps = None
    web_chunk.web = MagicMock(title="Review blog", uri="https://blog.example.com/kai-lane")

    _mock_genai(
        monkeypatch,
        [
            web_chunk,
            _maps_chunk("Kai Lane Coffee", "place-123", "https://maps.google.com/?cid=123"),
        ],
    )

    answer = maps_answer_service("quiet cafe near Raffles Place")

    assert "**Sources:**" in answer
    assert "[Review blog](https://blog.example.com/kai-lane)" in answer
    assert "**Places:**" in answer
    assert "(place_id: place-123)" in answer


def test_maps_answer_without_grounding_chunks(monkeypatch: pytest.MonkeyPatch):
    """No chunks — plain answer, no Places section scaffolding."""
    client = _mock_genai(monkeypatch, [])

    answer = maps_answer_service("what is the capital of France")

    assert answer == "Try Kai Lane Coffee — quiet, good pour-overs."
    assert "Places" not in answer

    # Non-location questions can pass through, but the tool is still declared.
    config = client.models.generate_content.call_args.kwargs["config"]
    assert any(t.google_maps is not None for t in config.tools)


def test_gemini_maps_answer_is_registered():
    """The tool must be in the registry so _register_all_tools() auto-exposes it on /mcp/sse."""
    assert "gemini_maps_answer" in list_tool_names()
    assert has_tool("gemini_maps_answer")
