from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.database.db import Base, engine
from app.mcp.server import MCPAuthMiddleware, mcp
from app.routers import all_routers
from app.utils.exceptions import http_exception_handler, unhandled_exception_handler
from app.utils.limiter import limiter
from app.utils.logging import setup_logging
from app.utils.response import APIResponse

# Setup logging before FastAPI instance
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="AI/LLM Playground",
    description="",
    version="1.0.0",
    lifespan=lifespan
)

# Add MCP middleware
app.add_middleware(MCPAuthMiddleware)

# Register custom exception handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_exception_handler)


# Include all routers
for router in all_routers:
    app.include_router(router)

# Mount MCP
mcp_app = mcp.http_app(transport="sse")
app.mount("/mcp", mcp_app)


@app.get("/api/v1/health", response_model=APIResponse)
async def health() -> APIResponse:  # type: ignore[type-arg]
    return APIResponse(
        data={"message": "Health check passed"}
    )


@app.get("/", response_model=APIResponse)
async def root() -> APIResponse:  # type: ignore[type-arg]
    return APIResponse(
        data={"message": "App is running"}
    )