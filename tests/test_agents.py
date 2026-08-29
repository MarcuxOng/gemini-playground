import pytest
from fastapi.testclient import TestClient


def test_list_agents_returns_200(client: TestClient, auth_headers):
    response = client.get("/api/v1/agents/list", headers=auth_headers)
    assert response.status_code == 200


def test_run_agent_returns_401_without_auth(client: TestClient):
    response = client.post("/api/v1/agents/run", json={"prompt": "hello", "preset": "coder"})
    assert response.status_code in [401, 422]


@pytest.mark.parametrize("bad_model", ["gpt-4", "../../etc/passwd", "claude-3"])
def test_run_agent_rejects_invalid_model(client: TestClient, auth_headers, bad_model: str):
    response = client.post(
        "/api/v1/agents/run",
        json={"model": bad_model, "prompt": "hello", "preset": "research"},
        headers=auth_headers,
    )
    assert response.status_code == 422


# ── Preset ↔ registry contract ────────────────────────────────────────────────
#
# Regression guard for the `knowledge` preset shipping a phantom `scrape_url`
# tool: `project_tools_to_langchain` raises ValueError for unknown names, so a
# preset naming a tool that isn't registered crashes at build time, not import
# time — nothing caught it until the agent was actually run.


def _preset_tool_names() -> list[tuple[str, str]]:
    """Every (preset_name, tool_name) pair declared by a preset's TOOLS list."""
    import importlib

    from app.agents import PRESETS

    pairs: list[tuple[str, str]] = []
    for preset_name in PRESETS:
        module = importlib.import_module(f"app.agents.presets.{preset_name}")
        for tool_name in getattr(module, "TOOLS", []):
            if isinstance(tool_name, str):
                pairs.append((preset_name, tool_name))
    return pairs


@pytest.mark.parametrize("preset_name,tool_name", _preset_tool_names())
def test_preset_tools_exist_in_registry(preset_name: str, tool_name: str):
    """Each tool a preset declares must be registered, or the preset can't build."""
    from app.tools import has_tool, list_tool_names

    assert has_tool(tool_name), (
        f"Preset {preset_name!r} declares tool {tool_name!r}, which is not in the "
        f"registry. Registered tools: {sorted(list_tool_names())}"
    )


def test_every_preset_builds_its_tool_list():
    """The projection step itself must succeed for every preset."""
    import importlib

    from app.agents.base import project_tools_to_langchain
    from app.agents import PRESETS

    for preset_name in PRESETS:
        module = importlib.import_module(f"app.agents.presets.{preset_name}")
        projected = project_tools_to_langchain(getattr(module, "TOOLS", []))
        assert len(projected) == len(module.TOOLS)
