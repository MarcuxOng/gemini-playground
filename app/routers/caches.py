from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.database.db import get_db
from app.database.models import APIKey
from app.services import caches as caches_service
from app.services.gemini import resolve_attachments
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse
from app.utils.validators import ModelName

router = APIRouter(
    prefix="/api/v1/caches",
    tags=["Context Caches"],
    dependencies=[Depends(verify_api_key)],
)


def _validate_attachment_ids(v: list[str]) -> list[str]:
    for att in v:
        try:
            uuid.UUID(att)
        except ValueError:
            raise ValueError(f"Attachment must be a DB file UUID, got: {att!r}") from None
    return v


class CreateCacheInput(BaseModel):
    model: ModelName
    attachments: list[str] = []
    system_instruction: str | None = Field(default=None, max_length=32_000)
    display_name: str | None = Field(default=None, max_length=256)
    ttl: str = Field(default="3600s", pattern=r"^\d+s$")

    @field_validator("attachments")
    @classmethod
    def validate_attachments(cls, v: list[str]) -> list[str]:
        return _validate_attachment_ids(v)


class UpdateCacheInput(BaseModel):
    ttl: str = Field(..., pattern=r"^\d+s$")


@router.post("/", response_model=APIResponse)
@limiter.limit("10/minute")
async def create_cache(
    request: Request,
    body: CreateCacheInput,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    resolved = resolve_attachments(body.attachments, db, str(api_key.id))
    file_uris = [r["uri"] for r in resolved]
    mime_types = [r["mime_type"] for r in resolved]

    if not file_uris and not body.system_instruction:
        raise HTTPException(
            status_code=400,
            detail="At least one attachment or a system_instruction is required",
        )

    result = await run_in_threadpool(
        caches_service.create_context_cache,
        model=body.model,
        file_uris=file_uris,
        mime_types=mime_types,
        system_instruction=body.system_instruction,
        display_name=body.display_name,
        ttl=body.ttl,
    )
    return APIResponse(data=result)


@router.get("/", response_model=APIResponse)
@limiter.limit("20/minute")
async def list_caches(request: Request) -> APIResponse:  # type: ignore[type-arg]
    result = await run_in_threadpool(caches_service.list_caches)
    return APIResponse(data=result)


@router.get("/{cache_id:path}", response_model=APIResponse)
@limiter.limit("20/minute")
async def get_cache(request: Request, cache_id: str) -> APIResponse:  # type: ignore[type-arg]
    result = await run_in_threadpool(caches_service.get_cache, cache_id)
    return APIResponse(data=result)


@router.delete("/{cache_id:path}", response_model=APIResponse)
@limiter.limit("10/minute")
async def delete_cache(request: Request, cache_id: str) -> APIResponse:  # type: ignore[type-arg]
    await run_in_threadpool(caches_service.delete_cache, cache_id)
    return APIResponse(data={"deleted": cache_id})


@router.patch("/{cache_id:path}", response_model=APIResponse)
@limiter.limit("10/minute")
async def update_cache(
    request: Request,
    cache_id: str,
    body: UpdateCacheInput,
) -> APIResponse:  # type: ignore[type-arg]
    result = await run_in_threadpool(caches_service.update_cache_ttl, cache_id, body.ttl)
    return APIResponse(data=result)
