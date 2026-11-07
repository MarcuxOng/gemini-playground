"""gemini_ask_file — P1.1, the first Option B tool.

The service is tested through its real code path with only the genai client
mocked (no mocking the function under test). The delete-in-finally contract is
asserted on both the success and the failure path: a stateless tool that leaks
uploads is not stateless.
"""

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.gemini import ask_file_service
from app.tools import has_tool, list_tool_names


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    # A real PNG header so libmagic detects image/png rather than a text guess.
    p = tmp_path / "chart.png"
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return p


def _mock_genai(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Patch the module-level genai clients and the Files API upload/delete.

    The generate call goes through `generate_content_with_fallback`, whose
    clients live in `app.services.genai_calls` — patch those (as conftest does),
    not only `app.services.gemini`'s.
    """
    import app.services.genai_calls as calls_mod
    import app.services.gemini as mod

    response = MagicMock()
    response.text = "The chart shows a steady increase."
    response.candidates = []
    response.prompt_feedback = None

    client = MagicMock()
    client.models.generate_content.return_value = response
    uploaded = MagicMock()
    uploaded.name = f"files/{uuid.uuid4()}"
    uploaded.uri = f"https://generativelanguage.googleapis.com/v1beta/{uploaded.name}"
    client.files.upload.return_value = uploaded

    monkeypatch.setattr(mod, "client", client)
    monkeypatch.setattr(mod, "global_client", client)
    monkeypatch.setattr(calls_mod, "client", client)
    monkeypatch.setattr(calls_mod, "global_client", client)
    return client, uploaded


def test_ask_file_answers_from_the_file(png_file: Path, monkeypatch: pytest.MonkeyPatch):
    client, uploaded = _mock_genai(monkeypatch)

    answer = ask_file_service(str(png_file), "What does the chart show?")

    assert answer == "The chart shows a steady increase."
    kwargs = client.models.generate_content.call_args.kwargs
    contents = kwargs["contents"]
    # The Part references the upload (file_uri lives under file_data).
    assert contents[0].file_data.file_uri == uploaded.uri
    assert contents[0].file_data.mime_type == "image/png"
    assert contents[1] == "What does the chart show?"


def test_ask_file_deletes_upload_on_success(png_file: Path, monkeypatch: pytest.MonkeyPatch):
    client, uploaded = _mock_genai(monkeypatch)

    ask_file_service(str(png_file), "What does the chart show?")

    client.files.delete.assert_called_once_with(name=uploaded.name)


def test_ask_file_deletes_upload_on_model_failure(png_file: Path, monkeypatch: pytest.MonkeyPatch):
    """The finally-cleanup is the stateless contract — assert it survives an error."""
    import app.services.gemini as mod

    client, uploaded = _mock_genai(monkeypatch)
    client.models.generate_content.side_effect = RuntimeError("model exploded")

    with pytest.raises(RuntimeError):
        ask_file_service(str(png_file), "What does the chart show?")

    client.files.delete.assert_called_once_with(name=uploaded.name)


def test_ask_file_missing_file_raises_and_never_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client, _ = _mock_genai(monkeypatch)

    with pytest.raises(FileNotFoundError):
        ask_file_service(str(tmp_path / "nope.png"), "question")

    client.files.upload.assert_not_called()
    client.models.generate_content.assert_not_called()


def test_ask_file_rejects_disallowed_mime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """libmagic detects a .exe even when the extension says .png — extension lies are caught."""
    client, _ = _mock_genai(monkeypatch)
    # 'MZ' magic — libmagic detects application/x-dos_exe / x-msdos-executable.
    exe = tmp_path / "not-really.png"
    exe.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00")

    with pytest.raises(ValueError, match="Unsupported file type"):
        ask_file_service(str(exe), "question")

    client.files.upload.assert_not_called()


def test_gemini_ask_file_is_registered():
    """The tool must be in the registry so _register_all_tools() auto-exposes it on /mcp/sse."""
    assert "gemini_ask_file" in list_tool_names()
    assert has_tool("gemini_ask_file")
