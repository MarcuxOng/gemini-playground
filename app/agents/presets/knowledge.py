"""
A knowledge-focused RAG agent.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.agents.base import build_agent, merge_tools

CompiledGraph = Any

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a Knowledge Management Assistant with access to a private company database.
Your primary tool is `search_knowledge_base`, which allows you to retrieve context 
from private documents and internal knowledge.

Guidelines:
- When a user asks a question that seems specific to a particular context, company, 
  or private data, always start by using the `search_knowledge_base` tool.
- If you find relevant information, summarize it and present it as the primary answer.
- If the knowledge base does not contain the answer, explicitly state that you 
  could not find information on that topic in the internal database.
- Use your other tools (like web search or calculation) only if the user explicitly 
  requests external information or if it's necessary to supplement the internal data.
- Maintain a professional and helpful tone.
"""

# ── Factory ───────────────────────────────────────────────────────────────────

TOOLS = [
    "search_knowledge_base",
    "calculate",
    "scrape_url"
]


def build_knowledge_agent(
    model: str, 
    checkpointer: Any = None,
    extra_tools: list[BaseTool] | None = None,
) -> CompiledGraph:
    """
    Build and return a Knowledge/RAG ReAct agent.

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

