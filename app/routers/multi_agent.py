"""Routes for Gemini-native multi-agent systems.

All multi-agent endpoints require internal-key auth (x-internal-key header)
for server-to-server communication, or the standard API key for public endpoints.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, model_validator
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.agents import PRESETS
from app.database.db import get_db
from app.database.models import APIKey, Thread, ThreadMessage
from app.multi_agent.protocol import AgentMessage, agent_message_to_gemini_parts
from app.services.agents import AgentRunResponse, run_agent_service
from app.services.gemini import generate_thread_title
from app.utils.auth import verify_internal_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse
from app.utils.validators import ModelName

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Multi-Agent"],
)


class InvokeRequest(BaseModel):
    """Request body for the inter-agent invoke endpoint.

    Exactly one of ``target_preset`` or ``target_agent_id`` is required.
    """

    target_preset: str | None = None
    target_agent_id: str | None = None
    model: ModelName = "gemini-2.5-flash"
    message: AgentMessage
    thread_id: str | None = None

    @model_validator(mode="after")
    def check_target(self) -> InvokeRequest:
        if bool(self.target_preset) == bool(self.target_agent_id):
            raise ValueError("Exactly one of 'target_preset' or 'target_agent_id' is required.")
        return self


@router.post("/agents/invoke", response_model=APIResponse[AgentRunResponse])
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
    multimodal_parts: list[dict[str]] = []
    for part in agent_message_to_gemini_parts(body.message):
        if part.text:
            message_text += part.text
            multimodal_parts.append({"type": "text", "text": part.text})
        elif part.inline_data and part.inline_data.data:
            multimodal_parts.append(
                {
                    "type": "media",
                    "data": part.inline_data.data,
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

    # Handle threading
    thread_id = body.thread_id or str(uuid.uuid4())
    thread: Thread | None = None
    if body.thread_id:
        thread_query = db.query(Thread).filter(Thread.id == body.thread_id)
        thread = thread_query.first()
        if thread:
            thread_id = thread.id

    if not thread:
        title = await run_in_threadpool(generate_thread_title, message_text, target_model)
        thread = Thread(
            id=thread_id,
            owner_id="master",
            preset=preset_name,
            model=target_model,
            title=title,
        )
        db.add(thread)
        db.commit()
        db.refresh(thread)

    # Save human-style message
    human_msg = ThreadMessage(
        id=str(uuid.uuid4()),
        thread_id=thread.id,
        role="human",
        content=f"[MIAP from {body.message.sender_id}] {message_text}",
    )
    db.add(human_msg)
    db.commit()

    try:
        # Build synthetic AgentRunRequest for the target agent
        has_multimodal = any(p.get("type") == "media" for p in multimodal_parts)
        run_request = _AgentRunRequest(
            model=target_model,
            preset=body.target_preset,
            agent_id=body.target_agent_id,
            prompt=message_text,
            thread_id=thread.id,
            attachments=[],  # MIAP parts are passed inline, not via attachments
            multimodal_prompt=multimodal_parts if has_multimodal else None,
        )

        # Use the master API key identity for the synthetic request
        master_key = APIKey(id="master", name="Master Key")

        response = await run_agent_service(run_request, db, master_key, fastapi_request=request)

        # Save AI response
        ai_msg = ThreadMessage(
            id=str(uuid.uuid4()),
            thread_id=thread.id,
            role="ai",
            content=response.answer,
        )
        db.add(ai_msg)
        db.commit()

        return APIResponse(data=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in MIAP agent invoke")
        raise HTTPException(status_code=500, detail="Agent invocation failed.") from e
