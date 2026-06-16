"""
FastMCP server that exposes all registered tools to any MCP-compatible client.
Tools are pulled dynamically from the project's tool registry so there is
no duplication — adding a tool to the registry automatically exposes it here.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from limits import parse as parse_limit
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.config import settings
from app.database.db import SessionLocal
from app.tools import get_registry
from app.utils.auth import check_api_key

logger = logging.getLogger(__name__)
mcp = FastMCP(
    name="gemini-playground",
)

_mcp_rate_limit = parse_limit("60/minute")
_mcp_rate_limiter = FixedWindowRateLimiter(storage_from_string(settings.redis_url or "memory://"))


class MCPAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.startswith("/mcp"):
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            elif request.client and request.client.host:
                client_ip = request.client.host
            else:
                client_ip = "unknown"
            if not _mcp_rate_limiter.hit(_mcp_rate_limit, "mcp", client_ip):
                return JSONResponse(
                    {"error": "Rate limit exceeded. Max 60 requests/minute per IP."},
                    status_code=429,
                )

            api_key = request.headers.get("x-api-key")
            if not api_key:
                return JSONResponse({"error": "Unauthorized: Missing API Key"}, status_code=401)

            db = SessionLocal()
            try:
                authenticated = check_api_key(api_key, db)
            finally:
                db.close()

            if not authenticated:
                return JSONResponse({"error": "Unauthorized: Invalid API Key"}, status_code=401)

            return await call_next(request)
        return await call_next(request)


def _register_all_tools() -> None:
    """Dynamically register every tool in the project registry with FastMCP."""
    failures: list[str] = []
    for tool_name, entry in get_registry().items():
        fn = entry["fn"]

        try:
            mcp.tool(name=tool_name)(fn)
            logger.info(f"MCP: registered tool '{tool_name}'")
        except Exception:
            logger.exception("MCP: could not register tool '%s'", tool_name)
            failures.append(tool_name)

    if failures:
        raise RuntimeError(f"Failed to register MCP tools: {failures}")


_register_all_tools()
