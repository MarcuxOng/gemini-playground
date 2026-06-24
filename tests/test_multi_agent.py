"""Tests for the multi-agent router (Phase 8.6 — Multimodal Inter-Agent Protocol)."""

import base64

import pytest
from fastapi.testclient import TestClient


def test_agent_invoke_returns_403_without_internal_key(client: TestClient):
    """Public API key must not be accepted on the internal invoke endpoint."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "hello from agent A"}],
                "sender_id": "agent-a",
            },
        },
    )
    assert resp.status_code in (403, 422)  # 422 if FastAPI rejects missing header before dep


def test_agent_invoke_returns_403_with_bad_internal_key(client: TestClient):
    """Wrong internal key is rejected."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "hello"}],
                "sender_id": "agent-a",
            },
        },
        headers={"x-internal-key": "wrong-key"},
    )
    assert resp.status_code == 403


def test_agent_invoke_returns_200_with_valid_internal_key(
    client: TestClient, internal_auth_headers
):
    """Valid internal key + text part returns 200."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "hello from agent A"}],
                "sender_id": "agent-a",
            },
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "answer" in data["data"]
    assert data["data"]["answer"] == "mocked LLM response"


def test_agent_invoke_rejects_invalid_preset(client: TestClient, internal_auth_headers):
    """Unknown preset returns 400."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "nonexistent",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "test"}],
                "sender_id": "agent-a",
            },
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 400


def test_agent_invoke_rejects_missing_target(client: TestClient, internal_auth_headers):
    """Missing both target_preset and target_agent_id returns 422."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "test"}],
                "sender_id": "agent-a",
            },
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 422


def test_agent_invoke_rejects_both_targets(client: TestClient, internal_auth_headers):
    """Both target_preset and target_agent_id returns 422."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "target_agent_id": "some-id",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "test"}],
                "sender_id": "agent-a",
            },
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 422


def test_agent_invoke_with_metadata(client: TestClient, internal_auth_headers):
    """Message metadata is accepted."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "analyze this"}],
                "sender_id": "agent-a",
                "metadata": {"priority": "high", "source": "vision-agent"},
            },
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 200


def test_agent_invoke_with_thread_id(client: TestClient, internal_auth_headers):
    """Passing a thread_id works and returns the same thread_id."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "follow-up question"}],
                "sender_id": "agent-a",
            },
            "thread_id": "test-thread-001",
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"]["thread_id"] == "test-thread-001"


def test_agent_message_model_validation():
    """AgentMessage Pydantic model enforces min 1 part."""
    from app.multi_agent.protocol import AgentMessage

    with pytest.raises(Exception):
        AgentMessage(parts=[], sender_id="agent-a")


def test_agent_part_text_conversion():
    """Text AgentPart converts to Gemini Part correctly."""
    from app.multi_agent.protocol import AgentPart, agent_part_to_gemini_part

    ap = AgentPart(type="text", text="hello world")
    gp = agent_part_to_gemini_part(ap)
    assert gp.text == "hello world"


def test_agent_part_inline_data_conversion():
    """Inline data AgentPart converts to Gemini Part correctly."""
    from app.multi_agent.protocol import AgentPart, agent_part_to_gemini_part

    raw = b"fake-image-bytes"
    b64 = base64.b64encode(raw).decode()
    ap = AgentPart(type="inline_data", data=b64, mime_type="image/png")
    gp = agent_part_to_gemini_part(ap)
    assert gp.inline_data is not None
    assert gp.inline_data.data == raw
    assert gp.inline_data.mime_type == "image/png"


def test_agent_part_file_uri_conversion():
    """File URI AgentPart converts to Gemini Part correctly."""
    from app.multi_agent.protocol import AgentPart, agent_part_to_gemini_part

    ap = AgentPart(type="file_uri", file_uri="gs://bucket/file.pdf", mime_type="application/pdf")
    gp = agent_part_to_gemini_part(ap)
    assert gp.file_data is not None
    assert gp.file_data.file_uri == "gs://bucket/file.pdf"
    assert gp.file_data.mime_type == "application/pdf"


def test_agent_part_invalid_type():
    """Missing required field for declared type raises ValueError."""
    from app.multi_agent.protocol import AgentPart, agent_part_to_gemini_part

    ap = AgentPart(type="text", text=None)
    with pytest.raises(ValueError, match="requires 'text' field"):
        agent_part_to_gemini_part(ap)


def test_agent_part_missing_data_for_inline():
    """Inline data without 'data' raises ValueError."""
    from app.multi_agent.protocol import AgentPart, agent_part_to_gemini_part

    ap = AgentPart(type="inline_data", mime_type="image/png")
    with pytest.raises(ValueError, match="requires 'data' field"):
        agent_part_to_gemini_part(ap)


def test_agent_part_missing_mime_for_inline():
    """Inline data without 'mime_type' raises ValueError."""
    from app.multi_agent.protocol import AgentPart, agent_part_to_gemini_part

    ap = AgentPart(type="inline_data", data="dGVzdA==")
    with pytest.raises(ValueError, match="requires 'mime_type' field"):
        agent_part_to_gemini_part(ap)
