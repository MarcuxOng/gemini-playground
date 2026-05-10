from __future__ import annotations

from app.agents.base import build_agent
from app.agents.presets.analyst import build_analyst_agent
from app.agents.presets.coder import build_coder_agent
from app.agents.presets.knowledge import build_knowledge_agent
from app.agents.presets.research import build_research_agent
from app.agents.runner import AgentConfig, run_once

__all__ = [
    "build_agent",
    "run_once",
    "AgentConfig",
    "build_coder_agent",
    "build_research_agent",
    "build_analyst_agent",
    "build_knowledge_agent",
]

# DO NOT add a "supervisor" or other multi-agent preset here.
# Multi-agent orchestration is cut — see ROADMAP §4 "Explicitly Cut".
# Single-agent presets are fine to add (e.g. "operator" for Computer Use later).
PRESETS = {
    "coder": build_coder_agent,
    "research": build_research_agent,
    "analyst": build_analyst_agent,
    "knowledge": build_knowledge_agent,
}
