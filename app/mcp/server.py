"""
FastMCP server that exposes all registered tools to any MCP-compatible client.
Tools are pulled dynamically from the project's tool registry so there is
no duplication — adding a tool to the registry automatically exposes it here.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.database.db import SessionLocal
from app.tools import get_registry
from app.utils.auth import check_api_key

logger = logging.getLogger(__name__)
mcp = FastMCP(
    name="ai-llm-playground",
    instructions="An AI platform exposing tools",
)


class MCPAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # FastMCP SSE transport passes through HTTP headers
        if request.url.path.startswith("/mcp"):
            api_key = request.headers.get("x-api-key")
            if not api_key:
                return JSONResponse({"error": "Unauthorized: Missing API Key"}, status_code=401)

            db = SessionLocal()
            try:
                if check_api_key(api_key, db):
                    return await call_next(request)
            finally:
                db.close()

            return JSONResponse({"error": "Unauthorized: Invalid API Key"}, status_code=401)
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
