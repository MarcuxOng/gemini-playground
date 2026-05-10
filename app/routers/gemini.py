from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.gemini import (
    gemini_service, 
    list_gemini_models, 
    tools_service,
    gemini_stream_service
)
from app.utils.auth import verify_api_key
from app.utils.response import APIResponse
from app.utils.limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/gemini", 
    tags=["Gemini"],
    dependencies=[Depends(verify_api_key)]
)


class ProviderInput(BaseModel):
    model: str
    prompt: str


@router.get("/models", response_model=APIResponse)
async def get_gemini_model() -> APIResponse:  # type: ignore[type-arg]
    logger.info("Fetching available Gemini models")
    models = list_gemini_models()
    return APIResponse(data=models)


@router.post("/", response_model=APIResponse)
@limiter.limit("30/minute")
async def gemini(
    request: Request, 
    body: ProviderInput
) -> APIResponse:  # type: ignore[type-arg]
    logger.info(f"Calling Gemini API with model: {body.model}, prompt: {body.prompt}")
    response = gemini_service(
        model=body.model,
        prompt=body.prompt
    )

    return APIResponse(data=response)


@router.post("/tools", response_model=APIResponse)
@limiter.limit("10/minute")
async def tools(
    request: Request, 
    body: ProviderInput
) -> APIResponse:  # type: ignore[type-arg]
    """
    Gemini with tool calling support.
    """
    logger.info(f"Calling Gemini tools with model: {body.model}, prompt: {body.prompt}")
    response = tools_service(
        model=body.model,
        prompt=body.prompt
    )

    return APIResponse(data=response)


@router.post("/stream", response_class=StreamingResponse)
@limiter.limit("20/minute")
async def gemini_stream(
    request: Request, 
    body: ProviderInput
) -> StreamingResponse:
    """
    Stream Gemini response
    """
    logger.info(f"Starting Gemini stream with model: {body.model}, prompt: {body.prompt}")
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk in gemini_stream_service(
                body.model, 
                body.prompt
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'thread_id': None})}\n\n"
        except Exception:
            logger.exception("Error in Gemini stream")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Stream failed'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'thread_id': None})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")