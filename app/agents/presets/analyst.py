"""
An analyst-focused ReAct agent.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agents.base import build_agent, merge_tools

CompiledGraph = Any

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a financial analyst with access to the stock market and cryptocurrency data.

Guidelines:
- Always search for facts rather than relying on memory when recency matters.
- Fetch the actual page when a search snippet is insufficient.
- When answering, cite where information came from.
- Be thorough but concise — summarise long pages rather than quoting them wholesale.
"""

# ── Factory ───────────────────────────────────────────────────────────────────
TOOLS = [
    "get_stock_price",
    "get_crypto_price",
]


def build_analyst_agent(
    model: str,
    checkpointer: Any = None,
    extra_tools: list[BaseTool] | None = None,
    cached_content: str | None = None,
    max_output_tokens: int | None = None,
) -> CompiledGraph:
    """
    Build and return an analyst ReAct agent.

    Args:
        model: Model name.
        checkpointer: Optional LangGraph checkpointer.
        extra_tools: Optional additional LangChain tools.
        cached_content: Optional Gemini context cache ID.
        max_output_tokens: Optional max output tokens for generation.

    Returns:
        A compiled LangGraph agent.
    """
    combined_tools: list[str | BaseTool] = merge_tools(TOOLS, extra_tools)
    res = build_agent(
        tools=combined_tools,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        checkpointer=checkpointer,
        cached_content=cached_content,
        max_output_tokens=max_output_tokens,
    )
    return res
