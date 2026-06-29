"""Parallel reasoning engine — N Flash instances + Pro synthesis judge.

Dispatches the same problem to N Gemini Flash instances simultaneously,
each with a different perspective system prompt. A single Pro judge
synthesises the distinct outputs into one robust response.

Showcases Flash's speed advantage via ``asyncio.gather()`` and Pro's
superior reasoning for the synthesis step.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.config import default_model, eval_max_tokens, eval_model
from app.services.gemini import gemini_service, structured_service

logger = logging.getLogger(__name__)

DEFAULT_PERSPECTIVES = [
    "security engineer",
    "UX designer",
    "performance engineer",
    "product manager",
]

_WORKER_SYSTEM_PROMPT = "You are a {perspective}. {prompt}"

_JUDGE_PROMPT = """\
You are a synthesis judge. Combine the following expert perspectives into one definitive answer.

ORIGINAL QUESTION:
{prompt}

EXPERT PERSPECTIVES:
{perspectives_text}

Instructions:
1. Identify areas of agreement across perspectives — these are the strongest signals.
2. When perspectives disagree, explain the tension and adopt the most defensible position.
3. Flag any perspective that appears to misunderstand the question.
4. Produce a single, clear, actionable answer.

Respond with JSON matching the schema."""

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "answer": {
            "type": "STRING",
            "description": "The synthesised definitive answer.",
        },
        "reasoning": {
            "type": "STRING",
            "description": "How the perspectives were weighed and combined.",
        },
        "consensus": {
            "type": "BOOLEAN",
            "description": "True if the perspectives broadly agreed, False if significant disagreement remains.",
        },
    },
    "required": ["answer", "reasoning", "consensus"],
}

_PARTIAL_RESULTS_WARNING = (
    "(Note: {n_failed} of {n_total} worker{plural} failed to respond. "
    "The synthesis is based on partial results.)"
)


class ConsensusResult:
    """Immutable result container for a consensus run."""

    def __init__(
        self,
        answer: str,
        reasoning: str,
        consensus_reached: bool,
        perspectives: list[dict[str, str]],
        failed: int,
    ) -> None:
        self.answer = answer
        self.reasoning = reasoning
        self.consensus_reached = consensus_reached
        self.perspectives = perspectives
        self.failed = failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
            "consensus_reached": self.consensus_reached,
            "perspectives": self.perspectives,
            "failed_workers": self.failed,
        }


async def _run_worker(
    prompt: str,
    perspective: str,
    model: str,
    fastapi_request: Any = None,
) -> dict[str, str]:
    """Run a single perspective worker offloading the sync call to a thread."""
    worker_prompt = _WORKER_SYSTEM_PROMPT.format(perspective=perspective, prompt=prompt)
    response = await run_in_threadpool(
        gemini_service, model, worker_prompt, None, None, None, None, None, fastapi_request
    )
    return {"perspective": perspective, "response": str(response)}


def _format_perspectives(results: list[dict[str, str]]) -> str:
    """Format worker results for the judge prompt."""
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['perspective']}:\n{r['response']}")
    return "\n\n".join(parts)


async def run_consensus(
    prompt: str,
    model: str = default_model,
    perspectives: list[str] | None = None,
    judge_model: str = eval_model,
    max_output_tokens: int = eval_max_tokens,
    fastapi_request: Any = None,
) -> ConsensusResult:
    """Run the parallel reasoning engine.

    Args:
        prompt: The question or task to reason about.
        model: Gemini model for the N parallel workers.
        perspectives: List of expert role/system-prompt prefixes.
        judge_model: Gemini model for the synthesis judge (should be a Pro variant).
        max_output_tokens: Max tokens for the judge synthesis (default from eval_max_output_tokens).
        fastapi_request: Optional FastAPI Request for token tracking on request.state.

    Returns:
        ConsensusResult with the synthesised answer and per-perspective outputs.
    """
    if perspectives is None:
        perspectives = DEFAULT_PERSPECTIVES
    if not perspectives:
        raise ValueError("At least one perspective is required")

    logger.info(
        "Dispatching consensus run: prompt_len=%d perspectives=%d model=%s judge=%s",
        len(prompt),
        len(perspectives),
        model,
        judge_model,
    )

    worker_tasks = [_run_worker(prompt, p, model, fastapi_request) for p in perspectives]
    gathered = await asyncio.gather(*worker_tasks, return_exceptions=True)

    results: list[dict[str, str]] = []
    failed = 0
    for i, item in enumerate(gathered):
        if isinstance(item, Exception):
            logger.warning("Worker '%s' failed: %s", perspectives[i], item)
            failed += 1
        else:
            results.append(item)  # type: ignore[arg-type]

    perspectives_text = _format_perspectives(results)

    if failed == len(perspectives):
        raise RuntimeError("All consensus workers failed — cannot synthesise.")

    judge_prompt = _JUDGE_PROMPT.format(
        prompt=prompt,
        perspectives_text=perspectives_text,
    )

    try:
        judge_result = await run_in_threadpool(
            structured_service,
            judge_model,
            judge_prompt,
            _JUDGE_SCHEMA,
            fastapi_request,
            max_output_tokens,
        )
    except Exception as exc:
        logger.error("Consensus judge failed: %s", exc)
        raise RuntimeError(f"Consensus judge failed: {exc}") from exc

    answer = str(judge_result.get("answer", ""))
    reasoning = str(judge_result.get("reasoning", ""))
    consensus_reached = bool(judge_result.get("consensus", False))

    if failed > 0:
        plural = "s" if failed != 1 else ""
        note = _PARTIAL_RESULTS_WARNING.format(
            n_failed=failed, n_total=len(perspectives), plural=plural
        )
        answer = f"{note}\n\n{answer}"

    return ConsensusResult(
        answer=answer,
        reasoning=reasoning,
        consensus_reached=consensus_reached,
        perspectives=results,
        failed=failed,
    )
