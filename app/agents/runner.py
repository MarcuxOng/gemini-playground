"""
Agent runner — executes a compiled LangGraph agent and returns the answer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_once(
    agent: Any,
    question: str | list[Any],
    lg_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, int]]:
    """
    Send a single question to the agent and return its final answer.

    Args:
        agent:    A compiled LangGraph ReAct graph.
        question: The user's input message.
        lg_config:  Optional LangGraph config (e.g. for thread_id).

    Returns:
        Tuple of (answer string, token_usage dict with input_tokens / output_tokens).
    """
    try:
        if not question or (isinstance(question, str) and not question.strip()):
            raise ValueError("Question cannot be empty.")

        try:
            result = agent.invoke({"messages": [("human", question)]}, config=lg_config)
        except Exception as e:
            logger.exception("Agent invocation failed")
            raise RuntimeError(f"Agent invocation failed: {type(e).__name__}: {e}") from e

        content = result["messages"][-1].content

        input_tokens = 0
        output_tokens = 0
        for msg in result.get("messages", []):
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                um = msg.usage_metadata
                if hasattr(um, "get"):
                    input_tokens += int(um.get("input_tokens", 0))
                    output_tokens += int(um.get("output_tokens", 0))

        token_usage: dict[str, int] = {"input_tokens": input_tokens, "output_tokens": output_tokens}

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif isinstance(part, str):
                    text_parts.append(part)
            answer = "".join(text_parts)
        else:
            answer = str(content)

        return answer, token_usage
    except Exception as e:
        logger.error(f"Error in run_once: {e}")
        raise
