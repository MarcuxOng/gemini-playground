"""
FastMCP server that exposes all registered tools to any MCP-compatible client.
Tools are pulled dynamically from the project's tool registry so there is
no duplication — adding a tool to the registry automatically exposes it here.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from limits import parse as parse_limit
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.database.db import SessionLocal
from app.tools import get_registry
from app.utils.auth import check_api_key
from app.utils.limiter import limiter_hit

logger = logging.getLogger(__name__)
mcp = FastMCP(
    name="gemini-playground",
)

_mcp_rate_limit = parse_limit("60/minute")


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
            # Degrades to in-memory counters rather than 500ing if the store is
            # down (Decision 1). This gate runs before the API key check, so an
            # unhandled storage error here would close the MCP surface entirely.
            if not limiter_hit(_mcp_rate_limit, "mcp", client_ip):
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
