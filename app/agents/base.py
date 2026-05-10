"""
Core graph builder. All agent presets call build_agent() to
get a compiled LangGraph ReAct graph wired to a provider-specific LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent

from app.services.llm import build_llm
from app.tools import _REGISTRY

CompiledGraph = Any

logger = logging.getLogger(__name__)


def project_tools_to_langchain(tools: Sequence[str | BaseTool]) -> list[BaseTool]:
    """
    Convert project-specific registered tools to LangChain-compatible BaseTools.
    Handles a mix of tool names (strings) and already initialized BaseTool objects.
    
    Args:
        tools: List of tool names (strings) OR LangChain BaseTool objects.
        
    Returns:
        List of LangChain BaseTool objects.
    """
    lc_tools: list[BaseTool] = []
    missing_tools: list[str] = []
    for tool in tools:
        if not isinstance(tool, str):
            # If it's already a LangChain tool, keep it as is
            lc_tools.append(tool)
            continue

        entry = _REGISTRY.get(tool)
        if not entry:
            missing_tools.append(tool)
            continue
        
        fn = entry["fn"]
        schema = entry["schema"]
        
        # Create a LangChain tool from the function
        lc_tool = StructuredTool.from_function(
            func=fn,
            name=tool,
            description=schema["function"]["description"]
        )
        lc_tools.append(lc_tool)
        
    if missing_tools:
        raise ValueError(f"Tool(s) not found in registry: {missing_tools}")
    return lc_tools


def merge_tools(base_tools: Sequence[str | BaseTool], extra_tools: list[BaseTool] | None = None) -> list[str | BaseTool]:
    """
    Consolidates base tools and optional extra tools into a single list.
    This is the central place to handle collision resolution or validation.
    """
    combined: list[str | BaseTool] = list(base_tools)
    if extra_tools:
        combined.extend(extra_tools)
    return combined


def build_agent(
    tools: Sequence[str | BaseTool],
    system_prompt: str,
    model: str,
    checkpointer: Any = None,
) -> CompiledGraph:
    """
    Build and return a compiled LangGraph ReAct agent.

    Args:
        tools:         List of LangChain tools AND/OR project tool names.
        system_prompt: System-level instruction injected at the start of every run.
        model:         Model name.

    Returns:
        A compiled LangGraph CompiledGraph ready to invoke.
    """
    try:
        # Ensure all tools are converted to LangChain BaseTool objects
        processed_tools = project_tools_to_langchain(tools)

        agent = create_react_agent(
            model=build_llm(model),
            tools=processed_tools,
            prompt=system_prompt,
            checkpointer=checkpointer,
        )
        return agent
    except Exception as e:
        logger.error(f"Error building agent: {e}")
        raise

