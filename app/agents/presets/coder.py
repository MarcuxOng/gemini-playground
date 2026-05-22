"""
A coding-focused ReAct agent.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agents.base import build_agent, merge_tools

CompiledGraph = Any

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a senior software engineer and coding assistant.
You can write code in Python, JavaScript, Go, Rust, C++, and more.

Guidelines:
- Write clean code with type hints.
- Use comments to explain complex logic.
- Follow best practices for code structure and design.
- Write maintainable and scalable code.
- Explain your reasoning briefly before presenting the final code.
"""

# ── Factory ───────────────────────────────────────────────────────────────────

TOOLS = [
    "calculate",
    "read_file",
    "write_file",
    "test_regex",
    "count_tokens",
]

# Code execution will be added in Phase 4.2 via Gemini's native code_execution tool.


def build_coder_agent(
    model: str,
    checkpointer: Any = None,
    extra_tools: list[BaseTool] | None = None,
) -> CompiledGraph:
    """
    Build and return a coding ReAct agent.

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
