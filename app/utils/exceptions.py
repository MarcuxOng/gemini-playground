from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.utils.response import APIResponse

logger = logging.getLogger(__name__)


async def safety_block_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    response: APIResponse[None] = APIResponse(
        success=False,
        error="Request blocked by safety filters",
        meta={"path": request.url.path, "status_code": 400},
    )
    return JSONResponse(status_code=400, content=response.model_dump())


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"HTTP exception for {request.url.path}: {exc}")
    response: APIResponse[None] = APIResponse(
        success=False,
        error=str(exc.detail),
        meta={"path": request.url.path, "status_code": exc.status_code},
    )
    return JSONResponse(
        status_code=exc.status_code, content=response.model_dump(), headers=exc.headers
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception for {request.url.path}: {exc}")
    response: APIResponse[None] = APIResponse(
        success=False,
        error="Internal server error",
        meta={"path": request.url.path, "status_code": 500},
    )
    return JSONResponse(status_code=500, content=response.model_dump())
