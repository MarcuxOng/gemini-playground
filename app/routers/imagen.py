from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.services.imagen import edit_image_service, generate_image_service
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.mime import validate_upload
from app.utils.models import BaseRequestModel
from app.utils.response import APIResponse
from app.utils.validators import ModelName

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/imagen", tags=["Imagen"], dependencies=[Depends(verify_api_key)])


class ImageGenerationRequest(BaseRequestModel):
    prompt: str
    model: ModelName = "imagen-4.0-generate-001"


class ImageResponse(BaseModel):
    urls: list[str]


@router.post("/generate", response_model=APIResponse[ImageResponse])
@limiter.limit("5/minute")
async def generate_image(
    request: Request,
    body: ImageGenerationRequest,
) -> APIResponse[ImageResponse]:
    """Generate images from a text prompt."""
    try:
        urls = await run_in_threadpool(generate_image_service, prompt=body.prompt, model=body.model)
        return APIResponse(data=ImageResponse(urls=urls))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail="Image generation failed") from e


@router.post("/edit", response_model=APIResponse[ImageResponse])
@limiter.limit("5/minute")
async def edit_image(
    request: Request,
    prompt: str,
    file: UploadFile = File(...),
    model: ModelName = "imagen-4.0-generate-001",
) -> APIResponse[ImageResponse]:
    """Edit an image based on a text prompt."""
    try:
        if file.size is not None and file.size > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
        max_size = 20 * 1024 * 1024
        content = b""
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            content += chunk
            if len(content) > max_size:
                raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
        mime_type = file.content_type or "application/octet-stream"
        try:
            validate_upload(content, mime_type)
        except ValueError as e:
            detail = str(e)
            raise HTTPException(status_code=415, detail=detail) from e
        urls = await run_in_threadpool(
            edit_image_service, prompt=prompt, base_image_bytes=content, model=model
        )
        return APIResponse(data=ImageResponse(urls=urls))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image editing failed")
        raise HTTPException(status_code=500, detail="Image editing failed") from e
