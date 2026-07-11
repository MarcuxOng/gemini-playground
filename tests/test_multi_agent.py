"""Tests for multi-agent systems (Phase 8.6 MIAP + Phase 8.7 A2A Discovery)."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


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


# ── Phase 8.7 — A2A Discovery Mesh ──────────────────────────────────────────────


class TestAgentCardEndpoint:
    """Tests for ``GET /.well-known/agent.json``."""

    def test_agent_card_returns_200(self, client: TestClient):
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data

    def test_agent_card_has_required_fields(self, client: TestClient):
        resp = client.get("/.well-known/agent.json")
        card = resp.json()
        assert "name" in card
        assert "description" in card
        assert "url" in card
        assert "version" in card
        assert "protocol" in card
        assert "capabilities" in card
        assert "default_model" in card
        assert "mcp_tools" in card
        assert card["protocol"] == "a2a/1.0"
        assert card["version"] == "1.0"

    def test_agent_card_capabilities_are_non_empty(self, client: TestClient):
        resp = client.get("/.well-known/agent.json")
        card = resp.json()
        assert len(card["capabilities"]) >= 1
        for cap in card["capabilities"]:
            assert "name" in cap
            assert "description" in cap
            assert "tools" in cap


class TestBuildAgentCard:
    """Tests for ``build_agent_card()``."""

    def test_build_agent_card_returns_valid_card(self):
        from app.multi_agent.a2a import AgentCard, build_agent_card

        card = build_agent_card("http://localhost:8000")
        assert isinstance(card, AgentCard)
        assert card.name == "Gemini Playground"
        assert card.url == "http://localhost:8000"
        assert card.protocol == "a2a/1.0"

    def test_build_agent_card_strips_trailing_slash(self):
        from app.multi_agent.a2a import build_agent_card

        card = build_agent_card("http://localhost:8000/")
        assert card.url == "http://localhost:8000"

    def test_build_agent_card_includes_all_presets(self):
        from app.multi_agent.a2a import build_agent_card

        card = build_agent_card("http://localhost:8000")
        preset_names = {c.name for c in card.capabilities}
        assert preset_names >= {"research", "coder", "analyst", "knowledge"}

    def test_build_agent_card_capabilities_have_tools(self):
        from app.multi_agent.a2a import build_agent_card

        card = build_agent_card("http://localhost:8000")
        for cap in card.capabilities:
            assert len(cap.tools) > 0, f"Capability '{cap.name}' has no tools"

    def test_build_agent_card_sets_default_model(self):
        from app.multi_agent.a2a import build_agent_card

        card = build_agent_card("http://localhost:8000", default_model="gemini-2.5-pro")
        assert card.default_model == "gemini-2.5-pro"

    def test_build_agent_card_invoke_url(self):
        from app.multi_agent.a2a import build_agent_card

        card = build_agent_card("http://localhost:8000")
        assert card.invoke_url == "http://localhost:8000/api/v1/agents/invoke"


class TestAgentCardValidation:
    """Tests for ``AgentCard`` Pydantic model validation of incoming cards."""

    def test_validate_minimal_card(self):
        from app.multi_agent.a2a import AgentCard

        card = AgentCard(name="Test", description="A test agent", url="http://test.local")
        assert card.name == "Test"
        assert card.protocol == "a2a/1.0"
        assert card.capabilities == []

    def test_validate_full_card(self):
        from app.multi_agent.a2a import A2ACapability, AgentCard

        card = AgentCard(
            name="Peer Agent",
            description="External peer",
            url="https://peer.example.com",
            capabilities=[
                A2ACapability(
                    name="search",
                    description="Web search agent",
                    tools=["google_search", "get_weather"],
                )
            ],
        )
        assert len(card.capabilities) == 1
        assert card.capabilities[0].name == "search"

    def test_validate_card_missing_url_fails(self):
        from app.multi_agent.a2a import AgentCard

        with pytest.raises(ValueError):
            AgentCard(name="Test", description="Missing URL")


class TestA2ARouter:
    """Tests for ``A2ARouter`` routing logic."""

    @staticmethod
    def _make_mock_llm(response: str):
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(content=response)
        return mock

    @pytest.mark.asyncio
    async def test_route_three_scenarios(self):
        """Verify 3/3 routing scenarios select the correct agent."""
        from app.multi_agent.a2a import A2ACapability, A2ARouter, AgentCard, build_agent_card

        host_card = build_agent_card("http://localhost:8000")

        scenarios = [
            ("What's the weather in Tokyo?", "research", 0),
            ("Write a Python script to sort a list", "coder", 1),
            ("What's the current price of AAPL stock?", "analyst", 2),
        ]

        for task, expected_cap, idx in scenarios:
            mock_llm = self._make_mock_llm(f"[{idx}]")
            with patch("app.multi_agent.a2a.build_llm", return_value=mock_llm):
                # Add a peer to force multi-candidate routing through the LLM
                peer_card = AgentCard(
                    name="Peer Test",
                    description="External peer",
                    url="https://peer.example.com",
                    capabilities=[
                        A2ACapability(name="dummy", description="Dummy peer", tools=["calculate"]),
                    ],
                )
                router = A2ARouter(host_card=host_card)
                router._peers["https://peer.example.com"] = peer_card

                url, card = await router.route(task)

                cap_names = [c.name for c in card.capabilities]
                assert expected_cap in cap_names, (
                    f"Task '{task}' expected capability '{expected_cap}' in {cap_names}"
                )
                assert url == "host"

    @pytest.mark.asyncio
    async def test_route_raises_value_error_when_no_match(self):
        from app.multi_agent.a2a import A2ACapability, A2ARouter, AgentCard, build_agent_card

        host_card = build_agent_card("http://localhost:8000")

        mock_llm = self._make_mock_llm("nonexistent_agent")
        with patch("app.multi_agent.a2a.build_llm", return_value=mock_llm):
            # Add a peer to force multi-candidate routing through the LLM
            peer_card = AgentCard(
                name="Peer Test",
                description="External peer",
                url="https://peer.example.com",
                capabilities=[
                    A2ACapability(name="dummy", description="Dummy peer", tools=["calculate"]),
                ],
            )
            router = A2ARouter(host_card=host_card)
            router._peers["https://peer.example.com"] = peer_card

            with pytest.raises(ValueError, match="no candidate agent matches"):
                await router.route("Some random task")

    @pytest.mark.asyncio
    async def test_route_raises_value_error_with_no_candidates(self):
        from app.multi_agent.a2a import A2ARouter

        router = A2ARouter(host_card=None)
        with pytest.raises(ValueError, match="No agents available"):
            await router.route("Some task")

    @pytest.mark.asyncio
    async def test_route_single_candidate_is_fast_path(self):
        from app.multi_agent.a2a import A2ARouter, build_agent_card

        host_card = build_agent_card("http://localhost:8000")
        router = A2ARouter(host_card=host_card)

        with patch("app.multi_agent.a2a.build_llm") as mock_build:
            url, card = await router.route("Any task")
            assert url == "host"
            mock_build.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_parses_valid_cards(self):
        from app.multi_agent.a2a import A2ACapability, A2ARouter, AgentCard

        peer_card = AgentCard(
            name="Peer",
            description="Peer agent",
            url="https://peer1.example.com",
            capabilities=[A2ACapability(name="search", description="Search agent", tools=["google_search"])],
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = peer_card.model_dump()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("app.multi_agent.a2a.httpx.AsyncClient", return_value=mock_client),
            patch("app.multi_agent.a2a._check_peer_hostname", return_value=None) as mock_check_hostname,
        ):
            router = A2ARouter()
            discovered = await router.discover(["https://peer1.example.com"])
            assert discovered == ["https://peer1.example.com"]
            assert "https://peer1.example.com" in router.known_peers
            mock_check_hostname.assert_called_once_with("peer1.example.com")

    def test_known_peers_returns_copy(self):
        from app.multi_agent.a2a import A2ARouter, build_agent_card

        host_card = build_agent_card("http://localhost:8000")
        router = A2ARouter(host_card=host_card)
        peers = router.known_peers
        assert peers == {}
        peers["injected"] = host_card
        assert "injected" not in router.known_peers


class TestA2ARouteEndpoint:
    """Tests for ``POST /api/v1/agents/a2a/route``."""

    def test_a2a_route_returns_401_without_auth(self, client: TestClient):
        resp = client.post(
            "/api/v1/agents/a2a/route",
            json={"task": "What is the weather?"},
        )
        assert resp.status_code in (401, 422)

    def test_a2a_route_returns_200_with_auth(self, client: TestClient, auth_headers):
        resp = client.post(
            "/api/v1/agents/a2a/route",
            json={"task": "What is the weather in Tokyo?"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "selected_url" in data["data"]
        assert "agent_name" in data["data"]
        assert "capabilities" in data["data"]
        assert "discovered_peers" in data["data"]
        assert "total_candidates" in data["data"]

    def test_a2a_route_task_passed_through(self, client: TestClient, auth_headers):
        resp = client.post(
            "/api/v1/agents/a2a/route",
            json={"task": "Find me stock prices"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["task"] == "Find me stock prices"

    def test_a2a_route_selected_url_is_host_when_no_peers(self, client: TestClient, auth_headers):
        resp = client.post(
            "/api/v1/agents/a2a/route",
            json={"task": "Code a sorting algorithm"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["selected_url"] == "host"
        assert data["discovered_peers"] == []
        assert data["total_candidates"] == 1

    def test_a2a_route_rejects_empty_task(self, client: TestClient, auth_headers):
        resp = client.post(
            "/api/v1/agents/a2a/route",
            json={"task": ""},
            headers=auth_headers,
        )
        assert resp.status_code == 422


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

        # Snapshot build_llm call count to isolate this test
        build_llm_before = gemini_svc.build_llm.call_count

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

        # Workers run via gemini_service → build_llm(model).invoke().
        # Verify the 2 workers each called build_llm with the worker model.
        worker_calls = gemini_svc.build_llm.call_args_list[build_llm_before:]
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
