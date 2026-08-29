"""Tests for multi-agent systems (Phase 8.6 MIAP + Phase 8.8 parallel reasoning)."""

import base64
from unittest.mock import MagicMock, patch

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


def test_agent_invoke_returns_new_thread_id_when_none_given(client: TestClient, internal_auth_headers):
    """When no thread_id is provided a new thread is created and returned."""
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
        headers=internal_auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"]["thread_id"] is not None


def test_agent_invoke_returns_404_for_nonexistent_thread_id(client: TestClient, internal_auth_headers):
    """A nonexistent thread_id returns 404."""
    resp = client.post(
        "/api/v1/agents/invoke",
        json={
            "target_preset": "research",
            "model": "gemini-2.5-flash",
            "message": {
                "parts": [{"type": "text", "text": "follow-up"}],
                "sender_id": "agent-a",
            },
            "thread_id": "nonexistent-thread-999",
        },
        headers=internal_auth_headers,
    )
    assert resp.status_code == 404


def test_agent_message_model_validation():
    """AgentMessage Pydantic model enforces min 1 part."""
    from pydantic import ValidationError

    from app.multi_agent.protocol import AgentMessage

    with pytest.raises(ValidationError):
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


# ── Phase 8.8 — Parallel Reasoning Engine ─────────────────────────────────────────


class TestConsensusEndpoint:
    """Tests for ``POST /api/v1/agents/consensus``."""

    @staticmethod
    def _patch_judge_response(mock_client, text=None):
        """Patch ``client.models.generate_content`` with a JSON response the judge can parse."""
        response = MagicMock()
        response.text = text or '{"answer": "synthesised answer", "reasoning": "weighed perspectives", "consensus": true}'
        response.candidates = []
        response.prompt_feedback = None
        return patch.object(mock_client.models, "generate_content", return_value=response)

    def test_consensus_returns_401_without_auth(self, client: TestClient):
        resp = client.post(
            "/api/v1/agents/consensus",
            json={"prompt": "What is the best language for systems programming?"},
        )
        assert resp.status_code == 401

    def test_consensus_returns_200_with_auth(self, client: TestClient, auth_headers, mock_gemini_client_global):
        with self._patch_judge_response(mock_gemini_client_global):
            resp = client.post(
                "/api/v1/agents/consensus",
                json={"prompt": "What is the best language for systems programming?"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "answer" in data["data"]
        assert "reasoning" in data["data"]
        assert "consensus_reached" in data["data"]
        assert "perspectives" in data["data"]
        assert "failed_workers" in data["data"]

    def test_consensus_returns_default_perspectives(self, client: TestClient, auth_headers, mock_gemini_client_global):
        with self._patch_judge_response(mock_gemini_client_global):
            resp = client.post(
                "/api/v1/agents/consensus",
                json={"prompt": "How should we structure a microservice?"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["perspectives"]) == 4  # default perspectives

    def test_consensus_accepts_custom_perspectives(self, client: TestClient, auth_headers, mock_gemini_client_global):
        with self._patch_judge_response(mock_gemini_client_global):
            resp = client.post(
                "/api/v1/agents/consensus",
                json={
                    "prompt": "How should we structure a microservice?",
                    "perspectives": ["backend engineer", "devops engineer"],
                },
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["perspectives"]) == 2
        assert data["perspectives"][0]["perspective"] == "backend engineer"

    def test_consensus_accepts_custom_models(self, client: TestClient, auth_headers, mock_gemini_client_global):
        import app.services.gemini as gemini_svc

        # Snapshot the builder call count to isolate this test
        build_llm_before = gemini_svc.build_llm_with_region_fallback.call_count

        with self._patch_judge_response(mock_gemini_client_global) as patched_gen:
            resp = client.post(
                "/api/v1/agents/consensus",
                json={
                    "prompt": "Explain the CAP theorem.",
                    "model": "gemini-2.5-flash",
                    "judge_model": "gemini-2.5-pro",
                    "perspectives": ["DBA", "SRE"],
                },
                headers=auth_headers,
            )
        assert resp.status_code == 200

        # Workers run via gemini_service → build_llm_with_region_fallback(model).invoke().
        # Verify the 2 workers each built an LLM for the worker model.
        worker_calls = gemini_svc.build_llm_with_region_fallback.call_args_list[
            build_llm_before:
        ]
        assert len(worker_calls) == 2
        for call in worker_calls:
            assert call[0][0] == "gemini-2.5-flash", f"worker model mismatch: {call[0]}"

        # Judge runs via structured_service → client.models.generate_content(model=judge_model).
        # Assert the patched mock captured the judge model.
        judge_call = patched_gen.call_args
        assert judge_call is not None, "judge model was never called"
        assert judge_call[1].get("model") == "gemini-2.5-pro", (
            f"judge model mismatch: {judge_call[1]}"
        )

    def test_consensus_rejects_empty_prompt(self, client: TestClient, auth_headers):
        resp = client.post(
            "/api/v1/agents/consensus",
            json={"prompt": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422
