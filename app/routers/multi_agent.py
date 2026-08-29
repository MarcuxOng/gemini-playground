"""Routes for Gemini-native multi-agent systems.

All multi-agent endpoints require internal-key auth (x-internal-key header)
for server-to-server communication, or the standard API key for public endpoints.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.agents import PRESETS
from app.config import default_model, settings
from app.database.db import get_db
from app.database.models import APIKey
from app.multi_agent.consensus import run_consensus
from app.multi_agent.protocol import AgentMessage, agent_message_to_gemini_parts
from app.services.agents import AgentRunResponse, run_agent_service
from app.services.caches import assert_cache_access
from app.utils.auth import verify_api_key, verify_internal_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse
from app.utils.sanitizer import sanitize_prompt
from app.utils.validators import ModelName

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Multi-Agent"],
)


class InvokeRequest(BaseModel):
    """Request body for the inter-agent invoke endpoint.

    Exactly one of ``target_preset`` or ``target_agent_id`` is required.
    """

    target_preset: str | None = None
    target_agent_id: str | None = None
    model: ModelName = default_model
    message: AgentMessage
    thread_id: str | None = None

    @model_validator(mode="after")
    def check_target(self) -> InvokeRequest:
        if bool(self.target_preset) == bool(self.target_agent_id):
            raise ValueError("Exactly one of 'target_preset' or 'target_agent_id' is required.")
        return self


class ConsensusRequest(BaseModel):
    """Request body for the parallel reasoning consensus endpoint."""

    prompt: str = Field(..., min_length=1, max_length=32_000)
    model: ModelName = default_model
    judge_model: ModelName = settings.gemini_eval_model
    # Each perspective is one more parallel Gemini call, so the list is capped:
    # an uncapped list turns a single rate-limited request into unbounded fan-out.
    perspectives: list[str] | None = Field(default=None, max_length=8)
    max_output_tokens: int = Field(settings.eval_max_output_tokens, ge=1, le=65_536)
    shared_cache_id: str | None = Field(default=None, max_length=256)


@router.post("/invoke", response_model=APIResponse[AgentRunResponse])
@limiter.limit("60/minute")
async def agent_invoke(
    request: Request,
    body: InvokeRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(verify_internal_key),
) -> APIResponse[AgentRunResponse]:
    """Server-to-server multimodal agent invocation.

    Accepts a MIAP ``AgentMessage`` (base64-encoded Parts, text, or file URIs)
    and dispatches it to a target agent.  Access is restricted to callers
    that present the ``x-internal-key`` header.

    Use this endpoint when one agent needs to pass raw multimodal data
    (screenshots, audio clips, PDFs) directly to another agent without
    lossy text transcription.
    """
    preset_name = ""
    target_model = str(body.model)

    # Build a synthetic AgentRunRequest to reuse the existing run_agent_service
    from app.services.agents import AgentRunRequest as _AgentRunRequest

    if body.target_agent_id:
        preset_name = f"invoke:{body.target_agent_id}"
    else:
        preset = str(body.target_preset)
        if preset not in PRESETS:
            raise HTTPException(
                status_code=400, detail=f"Invalid preset. Available: {list(PRESETS.keys())}"
            )
        preset_name = preset

    # Convert MIAP parts to text + multimodal content
    message_text = ""
    multimodal_parts: list[dict[str, Any]] = []
    for part in agent_message_to_gemini_parts(body.message):
        if part.text:
            message_text += part.text
            multimodal_parts.append({"type": "text", "text": part.text})
        elif part.inline_data and part.inline_data.data:
            multimodal_parts.append(
                {
                    "type": "media",
                    "data": base64.b64encode(part.inline_data.data).decode(),
                    "mime_type": part.inline_data.mime_type or "application/octet-stream",
                }
            )
        elif part.file_data and part.file_data.file_uri:
            file_uri = part.file_data.file_uri
            # Normalise short form "files/<name>" to full https URL
            if file_uri.startswith("files/"):
                file_uri = f"https://generativelanguage.googleapis.com/{file_uri}"
            multimodal_parts.append(
                {
                    "type": "media",
                    "file_uri": file_uri,
                    "mime_type": part.file_data.mime_type or "application/octet-stream",
                }
            )

    logger.info(
        "MIAP invoke: sender=%s target=%s parts=%d",
        body.message.sender_id,
        preset_name,
        len(body.message.parts),
    )

    message_text = sanitize_prompt(message_text) if message_text else message_text
    sender_prefix = f"[MIAP from {body.message.sender_id}] "
    has_multimodal = any(p.get("type") == "media" for p in multimodal_parts)
    if has_multimodal:
        multimodal_parts.insert(0, {"type": "text", "text": sender_prefix})

    try:
        run_request = _AgentRunRequest(
            model=target_model,
            preset=body.target_preset,
            agent_id=body.target_agent_id,
            prompt=f"{sender_prefix}{message_text}",
            thread_id=body.thread_id,
            attachments=[],
            multimodal_prompt=multimodal_parts if has_multimodal else None,
        )

        # Use the master API key identity for the synthetic request
        master_key = APIKey(id="master", name="Master Key")

        response = await run_agent_service(run_request, db, master_key, fastapi_request=request)

        return APIResponse(data=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in MIAP agent invoke")
        raise HTTPException(status_code=500, detail="Agent invocation failed.") from e


@router.post("/consensus", response_model=APIResponse)
@limiter.limit("10/minute")
async def agent_consensus(
    request: Request,
    body: ConsensusRequest,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    """Run the parallel reasoning engine.

    Dispatches the prompt to N Gemini Flash workers simultaneously,
    each with a different perspective system prompt. A Pro judge
    synthesises the outputs into one robust response.
    """
    prompt = sanitize_prompt(body.prompt)
    if body.shared_cache_id:
        assert_cache_access(db, body.shared_cache_id, str(api_key.id))
    try:
        request.state.model = f"{body.model}+{body.judge_model}"
        result = await run_consensus(
            prompt=prompt,
            model=str(body.model),
            perspectives=body.perspectives,
            judge_model=str(body.judge_model),
            max_output_tokens=body.max_output_tokens,
            fastapi_request=request,
            cache_id=body.shared_cache_id,
        )
        return APIResponse(data=result.to_dict())

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Consensus engine failed")
        raise HTTPException(status_code=500, detail="Consensus engine failed.") from exc
