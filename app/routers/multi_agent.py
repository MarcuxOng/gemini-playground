"""Routes for Gemini-native multi-agent systems.

All multi-agent endpoints require internal-key auth (x-internal-key header)
for server-to-server communication, or the standard API key for public endpoints.
"""

from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.agents import PRESETS
from app.config import default_model, eval_max_tokens, eval_model
from app.database.db import get_db
from app.database.models import APIKey
from app.multi_agent.a2a import A2ARouter, _check_peer_hostname, build_agent_card
from app.multi_agent.consensus import run_consensus
from app.multi_agent.protocol import AgentMessage, agent_message_to_gemini_parts
from app.services.agents import AgentRunResponse, run_agent_service
from app.utils.auth import verify_api_key, verify_internal_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse
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


class A2ARouteRequest(BaseModel):
    """Request body for the A2A routing endpoint."""

    task: str = Field(..., min_length=1, max_length=32_000)
    peer_urls: list[str] = Field(default_factory=list, max_length=20)
    model: ModelName = default_model
    shared_cache_id: str | None = Field(default=None, max_length=256)


class ConsensusRequest(BaseModel):
    """Request body for the parallel reasoning consensus endpoint."""

    prompt: str = Field(..., min_length=1, max_length=32_000)
    model: ModelName = default_model
    judge_model: ModelName = eval_model
    perspectives: list[str] | None = None
    max_output_tokens: int = Field(eval_max_tokens, ge=1, le=65_536)
    shared_cache_id: str | None = Field(default=None, max_length=256)


def _validate_peer_url(raw: str) -> None:
    """Reject peer URLs targeting internal or private addresses (SSRF guard)."""
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail=f"Unsupported scheme in peer URL: {raw}")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail=f"Invalid peer URL (no hostname): {raw}")
    try:
        _check_peer_hostname(hostname)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@router.post("/a2a/route", response_model=APIResponse)
@limiter.limit("30/minute")
async def a2a_route(
    request: Request,
    body: A2ARouteRequest,
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    """Route a task to the best-suited agent using A2A discovery.

    Discovers peer agents from the provided *peer_urls* (if any) and the host's
    own Agent Card, then uses Gemini to select the single best agent for the
    given task description — no hardcoded routing table.
    """
    base_url = str(request.base_url).rstrip("/")
    host_card = build_agent_card(base_url, default_model=str(body.model))

    router = A2ARouter(host_card=host_card)

    discovered: list[str] = []
    if body.peer_urls:
        for url in body.peer_urls:
            _validate_peer_url(url)
        discovered = await router.discover(body.peer_urls)

    try:
        selected_url, selected_card = await router.route(
            body.task, model=str(body.model), cache_id=body.shared_cache_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return APIResponse(
        data={
            "task": body.task,
            "selected_url": selected_url,
            "agent_name": selected_card.name,
            "capabilities": [c.model_dump() for c in selected_card.capabilities],
            "discovered_peers": discovered,
            "total_candidates": 1 + len(discovered),
        }
    )


@router.post("/consensus", response_model=APIResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def agent_consensus(
    request: Request,
    body: ConsensusRequest,
) -> APIResponse:  # type: ignore[type-arg]
    """Run the parallel reasoning engine.

    Dispatches the prompt to N Gemini Flash workers simultaneously,
    each with a different perspective system prompt. A Pro judge
    synthesises the outputs into one robust response.
    """
    try:
        request.state.model = f"{body.model}+{body.judge_model}"
        result = await run_consensus(
            prompt=body.prompt,
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
