"""
A research-focused ReAct agent.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agents.base import build_agent, merge_tools

CompiledGraph = Any

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a research assistant with access to Google Search and other tools.

Guidelines:
- Always search for facts using Google Search rather than relying on memory when recency matters.
- When answering, provide citations for your information.
- Be thorough but concise.
- If the user asks about current weather or time, use the appropriate tool.
"""

# ── Factory ───────────────────────────────────────────────────────────────────

TOOLS = [
    "google_search",
    "get_weather",
    "get_datetime_info",
    "get_wikipedia_summary",
    "get_youtube_transcript",
]


def build_research_agent(
    model: str,
    checkpointer: Any = None,
    extra_tools: list[BaseTool] | None = None,
) -> CompiledGraph:
    """
    Build and return a research ReAct agent.

    Args:
        model: Model name.
        checkpointer: Optional LangGraph checkpointer.
        extra_tools: Optional additional LangChain tools.

    Returns:
        A compiled LangGraph agent.
    """
    try:
        combined_tools: list[str | BaseTool] = merge_tools(TOOLS, extra_tools)
        res = build_agent(
            tools=combined_tools,
            system_prompt=SYSTEM_PROMPT,
            model=model,
            checkpointer=checkpointer,
        )
        return res
    except Exception:
        raise
