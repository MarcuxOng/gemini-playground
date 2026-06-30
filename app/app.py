from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.database.db import Base, engine
from app.mcp.server import MCPAuthMiddleware, mcp
from app.multi_agent.a2a import build_agent_card
from app.routers import all_routers
from app.services.gemini import SafetyBlockError
from app.utils.exceptions import (
    http_exception_handler,
    input_sanitization_exception_handler,
    safety_block_exception_handler,
    unhandled_exception_handler,
)
from app.utils.limiter import limiter
from app.utils.middleware import UsageLoggingMiddleware
from app.utils.observability import setup_observability
from app.utils.response import APIResponse
from app.utils.sanitizer import InputSanitizationError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    # Initialize observability
    setup_observability(app)
    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Gemini Playground", description="", version="1.0.0", lifespan=lifespan)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MCPAuthMiddleware)
app.add_middleware(UsageLoggingMiddleware)

# Register custom exception handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(SafetyBlockError, safety_block_exception_handler)
app.add_exception_handler(InputSanitizationError, input_sanitization_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# Include all routers
for router in all_routers:
    app.include_router(router)

# Mount MCP
mcp_app = mcp.http_app(transport="sse")
app.mount("/mcp", mcp_app)


@app.get("/api/v1/health", response_model=APIResponse)
async def health() -> APIResponse:  # type: ignore[type-arg]
    return APIResponse(data={"message": "Health check passed"})


@app.get("/.well-known/agent.json")
async def agent_card(request: Request) -> dict[str, object]:
    """A2A Agent Card — exposes hosted agent capabilities for peer discovery.

    Returns the raw Agent Card JSON (not wrapped in APIResponse) so that
    peer agents can discover capabilities directly — the A2A protocol
    expects the card document at the top level.
    """
    base_url = str(request.base_url).rstrip("/")
    card = build_agent_card(base_url)
    return card.model_dump()


@app.get("/", response_model=APIResponse)
async def root() -> APIResponse:  # type: ignore[type-arg]
    return APIResponse(data={"message": "App is running"})
