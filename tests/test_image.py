import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def _build_image_candidate(data: bytes, mime_type: str = "image/png") -> MagicMock:
    """Build a mock candidate with inline image data for extract_nano_banana_bytes."""
    inline = MagicMock()
    inline.data = data
    inline.mime_type = mime_type

    part = MagicMock()
    part.inline_data = inline

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    return candidate


# ── Model name validation ──────────────────────────────────────────────────


@pytest.mark.parametrize("bad_model", [
    "../../etc/passwd",
    "gpt-4",
    "openai/gpt-4",
    "",
])
def test_image_generate_rejects_invalid_model_names(client: TestClient, auth_headers, bad_model: str):
    response = client.post(
        "/api/v1/image/generate",
        json={"model": bad_model, "prompt": "a cat"},
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.parametrize("good_model", [
    "gemini-2.5-flash-image",
])
def test_image_generate_accepts_valid_model_names(
    client: TestClient,
    auth_headers,
    mock_gemini_client_global: MagicMock,
    good_model: str,
):
    mock_gemini_client_global.models.generate_content.reset_mock()

    mock_response = MagicMock()
    mock_response.candidates = [_build_image_candidate(b"fake-png-data")]
    mock_response.prompt_feedback = None
    mock_gemini_client_global.models.generate_content.return_value = mock_response

    response = client.post(
        "/api/v1/image/generate",
        json={"model": good_model, "prompt": "a cat"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert "urls" in data
    assert len(data["urls"]) == 1
    assert data["urls"][0].startswith("data:image/png;base64,")

    call_kwargs = mock_gemini_client_global.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == good_model
    config = call_kwargs["config"]
    assert "IMAGE" in config.response_modalities


def test_generate_image_returns_401_without_auth(client: TestClient):
    response = client.post(
        "/api/v1/image/generate",
        json={"model": "gemini-2.5-flash-image", "prompt": "a cat"},
    )
    assert response.status_code in [401, 422]


# ── Happy-path edit ────────────────────────────────────────────────────────


def test_edit_image_returns_url(client: TestClient, auth_headers, mock_gemini_client_global: MagicMock):
    mock_gemini_client_global.models.generate_content.reset_mock()

    mock_response = MagicMock()
    mock_response.candidates = [_build_image_candidate(b"fake-edited-png-data")]
    mock_response.prompt_feedback = None
    mock_gemini_client_global.models.generate_content.return_value = mock_response

    with patch("app.routers.image.validate_upload", return_value=None):
        response = client.post(
            "/api/v1/image/edit",
            params={"prompt": "Make it black and white"},
            files={"file": ("test.png", io.BytesIO(b"fake-image-bytes"), "image/png")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["urls"]) == 1
    assert data["urls"][0].startswith("data:image/png;base64,")

    call_kwargs = mock_gemini_client_global.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash-image"
    config = call_kwargs["config"]
    assert "IMAGE" in config.response_modalities


def test_edit_image_rejects_no_file(client: TestClient, auth_headers):
    response = client.post(
        "/api/v1/image/edit",
        params={"prompt": "Make it black and white"},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_edit_image_returns_401_without_auth(client: TestClient):
    response = client.post(
        "/api/v1/image/edit",
        params={"prompt": "edit"},
        files={"file": ("test.png", io.BytesIO(b"x"), "image/png")},
    )
    assert response.status_code in [401, 422]
