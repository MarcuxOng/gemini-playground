from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.agents.presets import KNOWN_PRESETS
from app.database.models import APIKey, EvalDataset, EvalRun
from app.services.agents import AgentRunRequest, run_agent_service
from app.services.gemini import structured_service

logger = logging.getLogger(__name__)

GRADER_PROMPT_TEMPLATE = """
You are an objective grader for an AI agent.
User Input: {input}
Expected Output: {expected}
Agent Output: {actual}

Determine if the Agent Output is correct and matches the essence of the Expected Output.
If the Expected Output is a specific value (like a number or date), the match must be exact.
If it is a text explanation, determine if the Agent Output covers the same key points.

Respond in JSON format:
{{
    "passed": boolean,
    "reason": "short explanation of why it passed or failed"
}}
"""

GRADER_SCHEMA = {
    "type": "object",
    "properties": {"passed": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["passed", "reason"],
}


async def run_eval(
    db: Session, dataset_id: str, agent_id_or_preset: str, model: str, api_key_id: str
) -> dict[str, Any]:
    """
    Runs an evaluation of an agent against a dataset.
    """
    dataset = db.query(EvalDataset).filter(EvalDataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    cases = list(dataset.cases)
    results = []
    passed_count = 0

    for case in cases:
        if not isinstance(case, dict) or "input" not in case or "expected" not in case:
            results.append(
                {
                    "input": str(case),
                    "expected": "",
                    "actual": "ERROR",
                    "passed": False,
                    "reason": "Invalid case format: missing required keys 'input' or 'expected'",
                }
            )
            continue
        user_input = case["input"]
        expected_output = case["expected"]
        attachments = case.get("attachments", [])
        if not isinstance(attachments, list):
            attachments = []

        # Run the agent
        # We create a mock APIKey object to satisfy the service signature if needed, but here we just need to pass the ID.

        mock_api_key = APIKey(id=api_key_id)
        run_request = AgentRunRequest(
            prompt=user_input,
            model=model,
            preset=agent_id_or_preset if agent_id_or_preset in KNOWN_PRESETS else None,
            agent_id=agent_id_or_preset if agent_id_or_preset not in KNOWN_PRESETS else None,
            attachments=attachments,
        )

        try:
            agent_response = await run_agent_service(run_request, db, mock_api_key)
            actual_output = agent_response.answer

            # Grade the output
            grader_prompt = GRADER_PROMPT_TEMPLATE.format(
                input=user_input, expected=expected_output, actual=actual_output
            )

            # Use structured_service as the grader
            grade = await run_in_threadpool(
                structured_service,
                model="gemini-3.1-pro-preview",
                prompt=grader_prompt,
                schema=GRADER_SCHEMA,
            )

            is_passed = grade.get("passed", False)
            if is_passed:
                passed_count += 1

            results.append(
                {
                    "input": user_input,
                    "expected": expected_output,
                    "actual": actual_output,
                    "passed": is_passed,
                    "reason": grade.get("reason", ""),
                }
            )

        except Exception as e:
            logger.error(f"Error in eval case for input '{user_input}': {e}")
            results.append(
                {
                    "input": user_input,
                    "expected": expected_output,
                    "actual": "ERROR",
                    "passed": False,
                    "reason": str(e),
                }
            )

    metrics = {
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "total": len(cases),
        "results": results,
    }

    # Save the run
    eval_run = EvalRun(
        id=str(uuid.uuid4()), dataset_id=dataset_id, agent_id=agent_id_or_preset, metrics=metrics
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)

    return {"run_id": eval_run.id, "metrics": metrics}
