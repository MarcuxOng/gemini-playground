"""gemini_run_code — P1.3, the third partial-overlap tool.

The overlap with a local code runner is partial by design: this runs in
Gemini's sandbox, never in-process (Hard Rule 6). The contract under test is
that the model's sandbox work — the code it wrote and the output it got — is
surfaced after the answer, in execution order, instead of being hidden.
"""

from unittest.mock import MagicMock

import pytest

from app.services.gemini import run_code_service
from app.tools import has_tool, list_tool_names


def _code_part(code: str) -> MagicMock:
    p = MagicMock()
    p.executable_code = MagicMock(code=code)
    p.code_execution_result = None
    return p


def _result_part(output: str) -> MagicMock:
    p = MagicMock()
    p.executable_code = None
    p.code_execution_result = MagicMock(output=output)
    return p


def _text_part() -> MagicMock:
    p = MagicMock()
    p.executable_code = None
    p.code_execution_result = None
    return p


def _mock_genai(monkeypatch: pytest.MonkeyPatch, parts: list[MagicMock], text: str | None):
    import app.services.gemini as mod
    import app.services.genai_calls as calls_mod

    candidate = MagicMock()
    candidate.content.parts = parts
    candidate.grounding_metadata = None
    response = MagicMock()
    response.text = text
    response.candidates = [candidate]
    response.prompt_feedback = None

    client = MagicMock()
    client.models.generate_content.return_value = response

    monkeypatch.setattr(mod, "client", client)
    monkeypatch.setattr(mod, "global_client", client)
    monkeypatch.setattr(calls_mod, "client", client)
    monkeypatch.setattr(calls_mod, "global_client", client)
    return client


def test_run_code_appends_code_and_output(monkeypatch: pytest.MonkeyPatch):
    client = _mock_genai(
        monkeypatch,
        [
            _code_part("print(2 + 2)"),
            _result_part("4"),
            _text_part(),
        ],
        "The answer is 4.",
    )

    answer = run_code_service("what is 2 + 2")

    assert answer.startswith("The answer is 4.")
    assert "**Code executed:**" in answer
    assert "```python" in answer
    assert "print(2 + 2)" in answer
    assert "4" in answer

    config = client.models.generate_content.call_args.kwargs["config"]
    assert any(t.code_execution is not None for t in config.tools), config.tools


def test_run_code_multiple_iterations_in_order(monkeypatch: pytest.MonkeyPatch):
    """Interleaved code/result parts must appear in the order they ran."""
    _mock_genai(
        monkeypatch,
        [
            _code_part("x = 2 + 2"),
            _result_part("4"),
            _code_part("print(x * 10)"),
            _result_part("40"),
        ],
        "The answer is 40.",
    )

    answer = run_code_service("what is 40")

    work = answer.split("**Code executed:**")[1]
    assert work.index("x = 2 + 2") < work.index("4\n")
    assert work.index("4\n") < work.index("print(x * 10)")
    assert work.index("print(x * 10)") < work.index("40")


def test_run_code_without_text_still_returns_the_work(monkeypatch: pytest.MonkeyPatch):
    """A run that ends in code output with no prose answer still returns the work."""
    _mock_genai(
        monkeypatch,
        [_code_part("print(sorted([3, 1, 2]))"), _result_part("[1, 2, 3]")],
        None,
    )

    answer = run_code_service("sort these numbers")

    assert answer.startswith("\n\n**Code executed:**")
    assert "print(sorted([3, 1, 2]))" in answer
    assert "[1, 2, 3]" in answer


def test_run_code_without_execution_parts_returns_plain_answer(
    monkeypatch: pytest.MonkeyPatch,
):
    """A task the model answers without running code — no scaffolding appended."""
    _mock_genai(monkeypatch, [_text_part()], "The answer is 4.")

    answer = run_code_service("what is 2 + 2")

    assert answer == "The answer is 4."
    assert "Code executed" not in answer


def test_gemini_run_code_is_registered():
    """The tool must be in the registry so _register_all_tools() auto-exposes it on /mcp/sse."""
    assert "gemini_run_code" in list_tool_names()
    assert has_tool("gemini_run_code")
