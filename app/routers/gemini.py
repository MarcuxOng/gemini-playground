from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.database.db import get_db
from app.database.models import APIKey
from app.services.gemini import (
    gemini_service,
    gemini_stream_service,
    list_gemini_models,
    structured_service,
)
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gemini", tags=["Gemini"], dependencies=[Depends(verify_api_key)])


class ProviderInput(BaseModel):
    model: str
    prompt: str
    attachments: list[str] = []
    native_tools: list[Literal["search", "code", "url"]] = []


class StructuredInput(BaseModel):
    model: str
    prompt: str
    response_schema: dict[str, Any]  # JSON Schema dict


@router.get("/models", response_model=APIResponse)
async def get_gemini_model() -> APIResponse:  # type: ignore[type-arg]
    logger.info("Fetching available Gemini models")
    models = list_gemini_models()
    return APIResponse(data=models)


@router.post("/", response_model=APIResponse)
@limiter.limit("30/minute")
async def gemini(
    request: Request,
    body: ProviderInput,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    logger.info(
        f"Calling Gemini API with model: {body.model}, prompt_len: {len(body.prompt)}, "
        f"attachments: {len(body.attachments)}, native_tools: {body.native_tools}"
    )
    logger.debug(f"Full prompt: {body.prompt!r}, attachment_ids: {body.attachments}")
    response = await run_in_threadpool(
        gemini_service,
        model=body.model,
        prompt=body.prompt,
        attachments=body.attachments,
        db=db,
        owner_id=str(api_key.id),
        native_tools=cast(list[str], body.native_tools),
    )

    return APIResponse(data=response)


@router.post("/structured", response_model=APIResponse)
@limiter.limit("20/minute")
async def gemini_structured(request: Request, body: StructuredInput) -> APIResponse:  # type: ignore[type-arg]
    logger.info(f"Calling Structured Gemini API with model: {body.model}, prompt: {body.prompt}")
    response = await run_in_threadpool(
        structured_service, model=body.model, prompt=body.prompt, schema=body.response_schema
    )

    return APIResponse(data=response)


@router.post("/stream", response_class=StreamingResponse)
@limiter.limit("20/minute")
async def gemini_stream(
    request: Request,
    body: ProviderInput,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> StreamingResponse:
    """
    Stream Gemini response
    """
    logger.info(
        f"Starting Gemini stream with model: {body.model}, prompt_len: {len(body.prompt)}, "
        f"attachments: {len(body.attachments)}, native_tools: {body.native_tools}"
    )
    logger.debug(f"Full prompt: {body.prompt!r}, attachment_ids: {body.attachments}")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in gemini_stream_service(
                body.model,
                body.prompt,
                attachments=body.attachments,
                db=db,
                owner_id=str(api_key.id),
                native_tools=cast(list[str], body.native_tools),
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'thread_id': None})}\n\n"
        except Exception:
            logger.exception("Error in Gemini stream")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Stream failed'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'thread_id': None})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
