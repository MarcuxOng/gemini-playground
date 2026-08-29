"""Token accounting: what gets counted, and once.

Two bugs lived here. ``run_once`` summed usage over the whole checkpointed
message list, so a thread's reported cost grew with every turn; and the raw
``genai`` path counted ``candidates_token_count`` only, dropping the thinking
tokens that Gemini 3 bills as output and that LangChain already includes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.runner import run_once
from app.services.gemini import _set_request_tokens


def _ai(text: str, inp: int, out: int) -> AIMessage:
    msg = AIMessage(content=text)
    msg.usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    return msg


def test_run_once_counts_only_the_current_turn():
    """A checkpointed thread replays its history; earlier turns are already paid for."""
    history = [
        HumanMessage(content="turn one"),
        _ai("answer one", 100, 50),
        HumanMessage(content="turn two"),
        _ai("answer two", 200, 70),
    ]
    agent = MagicMock()
    agent.invoke.return_value = {"messages": history}

    answer, usage = run_once(agent, "turn two")

    assert answer == "answer two"
    assert usage == {"input_tokens": 200, "output_tokens": 70}


def test_run_once_sums_multi_step_tool_calls_within_one_turn():
    """A ReAct turn makes several model calls; all of them belong to this turn."""
    messages = [
        HumanMessage(content="old"),
        _ai("old answer", 999, 999),
        HumanMessage(content="new"),
        _ai("calling a tool", 10, 5),
        _ai("final answer", 20, 8),
    ]
    agent = MagicMock()
    agent.invoke.return_value = {"messages": messages}

    _, usage = run_once(agent, "new")

    assert usage == {"input_tokens": 30, "output_tokens": 13}


def test_raw_usage_metadata_includes_thinking_tokens():
    """Gemini 3 spends most of a short answer on thinking; it is billed as output."""
    request = SimpleNamespace(state=SimpleNamespace())
    usage = SimpleNamespace(
        prompt_token_count=6, candidates_token_count=1, thoughts_token_count=81
    )

    _set_request_tokens(request, usage)

    assert request.state.input_tokens == 6
    assert request.state.output_tokens == 82


def test_raw_and_langchain_paths_agree():
    """The same call must not report different totals depending on which path served it."""
    raw_request = SimpleNamespace(state=SimpleNamespace())
    _set_request_tokens(
        raw_request,
        SimpleNamespace(prompt_token_count=6, candidates_token_count=1, thoughts_token_count=81),
    )

    lc_request = SimpleNamespace(state=SimpleNamespace())
    # LangChain already folds thinking into output_tokens.
    _set_request_tokens(lc_request, {"input_tokens": 6, "output_tokens": 82})

    assert raw_request.state.output_tokens == lc_request.state.output_tokens


def test_token_accounting_accumulates_across_calls():
    """Consensus makes N worker calls plus a judge call on one request."""
    request = SimpleNamespace(state=SimpleNamespace())
    for _ in range(3):
        _set_request_tokens(
            request,
            SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2),
        )
    assert request.state.input_tokens == 30
    assert request.state.output_tokens == 21
