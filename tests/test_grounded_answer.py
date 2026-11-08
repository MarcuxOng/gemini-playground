"""gemini_grounded_answer — P1.3, the first partial-overlap tool.

google_search already ships the grounded case end to end (answer + Sources
list). The work here is what a plain search client lacks: the grounding
supports — Gemini's own mapping of which answer segments are backed by which
sources. The contract under test is that those supports surface as a
**Citations:** block, each entry naming the chunk(s) it is grounded by.
"""

from unittest.mock import MagicMock

import pytest

from app.services.gemini import grounded_answer_service
from app.tools import has_tool, list_tool_names


def _web_chunk(title: str, uri: str) -> MagicMock:
    chunk = MagicMock()
    chunk.maps = None
    chunk.retrieved_context = None
    chunk.web = MagicMock(title=title, uri=uri)
    return chunk


def _support(text: str, indices: list[int]) -> MagicMock:
    s = MagicMock()
    s.segment.text = text
    s.segment.start_index = 0
    s.segment.end_index = len(text)
    s.grounding_chunk_indices = indices
    return s


def _mock_genai(
    monkeypatch: pytest.MonkeyPatch, chunks: list[MagicMock], supports: list[MagicMock]
):
    import app.services.gemini as mod
    import app.services.genai_calls as calls_mod

    candidate = MagicMock()
    candidate.grounding_metadata.grounding_chunks = chunks
    candidate.grounding_metadata.grounding_supports = supports
    response = MagicMock()
    response.text = "Python 3.13 is the latest stable release."
    response.candidates = [candidate]
    response.prompt_feedback = None

    client = MagicMock()
    client.models.generate_content.return_value = response

    monkeypatch.setattr(mod, "client", client)
    monkeypatch.setattr(mod, "global_client", client)
    monkeypatch.setattr(calls_mod, "client", client)
    monkeypatch.setattr(calls_mod, "global_client", client)
    return client


def test_grounded_answer_appends_sources_and_citations(
    monkeypatch: pytest.MonkeyPatch,
):
    chunks = [_web_chunk("Python release notes", "https://example.com/py")]
    supports = [_support("Python 3.13 is the latest stable release.", [0])]
    client = _mock_genai(monkeypatch, chunks, supports)

    answer = grounded_answer_service("latest python version")

    assert answer.startswith("Python 3.13 is the latest stable release.")
    assert "**Sources:**" in answer
    assert "[Python release notes](https://example.com/py)" in answer
    assert "**Citations:**" in answer
    assert (
        '"Python 3.13 is the latest stable release."'
        " — [Python release notes](https://example.com/py)" in answer
    )

    # The request must actually carry the google_search native tool.
    config = client.models.generate_content.call_args.kwargs["config"]
    assert any(t.google_search is not None for t in config.tools), config.tools


def test_grounded_answer_maps_supports_to_their_own_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    """A support's grounding_chunk_indices must resolve to the right chunk."""
    chunks = [
        _web_chunk("Python release notes", "https://example.com/py"),
        _web_chunk("PEP index", "https://example.com/pep"),
    ]
    supports = [_support("PEP 719 landed in 3.13.", [1])]
    _mock_genai(monkeypatch, chunks, supports)

    answer = grounded_answer_service("python 3.13 features")

    assert '"PEP 719 landed in 3.13." — [PEP index](https://example.com/pep)' in answer
    assert "[Python release notes]" not in answer.split("**Citations:**")[1]


def test_grounded_answer_truncates_long_segments(monkeypatch: pytest.MonkeyPatch):
    long_text = "This claim is backed by a source and the segment is quite long. " + "x" * 150
    _mock_genai(monkeypatch, [_web_chunk("S", "https://example.com/s")], [_support(long_text, [0])])

    answer = grounded_answer_service("q")

    citation_line = answer.split('"', 2)[1]
    assert len(citation_line) <= 120
    assert citation_line.endswith("...")


def test_grounded_answer_without_supports_has_no_citations_block(
    monkeypatch: pytest.MonkeyPatch,
):
    """Sources still surface; the Citations block only appears with supports."""
    _mock_genai(
        monkeypatch,
        [_web_chunk("Python release notes", "https://example.com/py")],
        [],
    )

    answer = grounded_answer_service("latest python version")

    assert "**Sources:**" in answer
    assert "**Citations:**" not in answer


def test_grounded_answer_support_with_no_indices_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
):
    """A support with no chunk indices cannot be cited — it is dropped, not emitted empty."""
    _mock_genai(
        monkeypatch,
        [_web_chunk("S", "https://example.com/s")],
        [_support("Claim with no backing index.", [])],
    )

    answer = grounded_answer_service("q")

    assert "**Citations:**" not in answer


def test_gemini_grounded_answer_is_registered():
    """The tool must be in the registry so _register_all_tools() auto-exposes it on /mcp/sse."""
    assert "gemini_grounded_answer" in list_tool_names()
    assert has_tool("gemini_grounded_answer")
