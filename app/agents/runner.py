"""
Handles executing a compiled LangGraph agent.
Supports single-shot runs and interactive REPL sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Configuration for a single agent run.

    Attributes:
        name:     Human-readable label shown in logs (e.g. "Research Agent").
        model:    Model string passed to build_agent().
        verbose:  If True, print tool calls and intermediate steps.
    """
    name: str = "Agent"
    model: str | None = None
    verbose: bool = True


def run_once(
    agent: Any, 
    question: str, 
    config: AgentConfig | None = None, 
    lg_config: dict[str, Any] | None = None
) -> str:
    """
    Send a single question to the agent and return its final answer.

    Args:
        agent:    A compiled LangGraph ReAct graph.
        question: The user's input message.
        config:   Optional AgentConfig for logging behaviour.
        lg_config:  Optional LangGraph config (e.g. for thread_id).

    Returns:
        The agent's final response as a plain string.
    """
    try:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        
        cfg = config or AgentConfig()
        if cfg.verbose:
            _print_divider(cfg.name, question)

        try:
            result = agent.invoke({"messages": [("human", question)]}, config=lg_config)
        except Exception as e:
            logger.exception("Agent invocation failed")
            raise RuntimeError("Agent invocation failed") from e

        content = result["messages"][-1].content


        # Handle Gemini/LangChain returning a list of content blocks instead of a string
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

        if cfg.verbose:
            _print_trace(result["messages"])
            print(f"\n  Agent: {answer}\n")

        return answer
    except Exception as e:
        logger.error(f"Error in run_once: {e}")
        raise


def run_interactive(agent: Any, config: AgentConfig | None = None) -> None:
    """
    Start an interactive REPL loop with the agent.
    Type 'quit', 'exit', or 'q' to stop.

    Args:
        agent:  A compiled LangGraph ReAct graph.
        config: Optional AgentConfig for logging behaviour.
    """
    cfg = config or AgentConfig()
    print(f"\n🤖  {cfg.name} ready  |  model: {cfg.model}\n    Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        run_once(agent, user_input, config=cfg)


# ── Private Helpers ───────────────────────────────────────────────────────────

def _print_divider(name: str, question: str) -> None:
    print(f"\n{'─' * 60}\n  [{name}]  {question}\n{'─' * 60}")


def _print_trace(messages: list[Any]) -> None:
    """Print tool calls and tool results from a message list."""
    for msg in messages[1:]:
        role = getattr(msg, "type", type(msg).__name__)
        if role == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"\n  ⚙  {tc['name']}({args})")
        elif role == "tool":
            preview = str(msg.content)[:200]
            suffix = "…" if len(str(msg.content)) > 200 else ""
            print(f"  ↳  {preview}{suffix}")
