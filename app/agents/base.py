"""
Core graph builder. All agent presets call build_agent() to
get a compiled LangGraph ReAct graph wired to a provider-specific LLM.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool, StructuredTool

from app.services.llm import build_llm
from app.tools import get_registry

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

        entry = get_registry().get(tool)
        if not entry:
            missing_tools.append(tool)
            continue

        fn = entry["fn"]
        schema = entry["schema"]

        # Create a LangChain tool from the function
        lc_tool = StructuredTool.from_function(
            func=fn, name=tool, description=schema["function"]["description"]
        )
        lc_tools.append(lc_tool)

    if missing_tools:
        raise ValueError(f"Tool(s) not found in registry: {missing_tools}")
    return lc_tools


def merge_tools(
    base_tools: Sequence[str | BaseTool], extra_tools: list[BaseTool] | None = None
) -> list[str | BaseTool]:
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
    native_tools: list[str] | None = None,
    cached_content: str | None = None,
    max_output_tokens: int | None = None,
) -> CompiledGraph:
    """
    Build and return a compiled LangGraph ReAct agent.

    Args:
        tools:         List of LangChain tools AND/OR project tool names.
        system_prompt: System-level instruction injected at the start of every run.
        model:         Model name.
        checkpointer:  Optional checkpointer for persistent state.
        native_tools:  Optional list of Gemini native tools ('search', 'code', 'url').
        cached_content: Optional Gemini context cache ID for shared context.
        max_output_tokens: Optional max output tokens for generation.

    Returns:
        A compiled LangGraph CompiledGraph ready to invoke.
    """
    try:
        if cached_content:
            if tools and any(tools):
                raise ValueError(
                    "cached_content cannot be combined with tools — tool declarations must be part of the cache."
                )
            if native_tools:
                raise ValueError(
                    "cached_content cannot be combined with native_tools — tool declarations must be part of the cache."
                )
            llm = build_llm(
                model, cached_content=cached_content, max_output_tokens=max_output_tokens
            )
            logger.info(
                "Using cached_content %s — system_instruction, tools, and tool_config "
                "must be part of the cache; skipping them in the request.",
                cached_content,
            )
            agent = create_agent(
                model=llm,
                tools=[],
                system_prompt=None,
                checkpointer=checkpointer,
            )
            return agent

        # Ensure all tools are converted to LangChain BaseTool objects
        processed_tools = project_tools_to_langchain(tools)

        # Build LLM and bind native tools if requested
        llm = build_llm(model, cached_content=cached_content, max_output_tokens=max_output_tokens)
        llm_with_tools: Any = llm

        # Bind native tools if requested
        if native_tools:
            lc_native_tools: list[dict[str, Any]] = []
            if "search" in native_tools:
                lc_native_tools.append({"google_search_retrieval": {}})
            if "code" in native_tools:
                # Code execution is handled differently in LangChain usually but for Gemini it can be a tool.
                lc_native_tools.append({"code_execution": {}})

            if lc_native_tools:
                llm_with_tools = llm.bind(tools=lc_native_tools)

        agent = create_agent(
            model=llm_with_tools,
            tools=processed_tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )
        return agent
    except Exception as e:
        logger.error(f"Error building agent: {e}")
        raise
