from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import APIKey, MCPServerConfig
from app.mcp.client import load_mcp_tools
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.models import BaseRequestModel
from app.utils.response import APIResponse

router = APIRouter(prefix="/api/v1/mcp-servers", tags=["MCP Servers"])


class MCPServerCreate(BaseRequestModel):
    name: str
    description: str | None = None
    transport: str | None = None  # "sse" or "stdio" (can be inferred)
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None


class MCPServerResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    transport: str | None = None
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.get("/", response_model=APIResponse[list[MCPServerResponse]])
@limiter.limit("30/minute")
async def list_mcp_servers(
    request: Request,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[list[MCPServerResponse]]:
    query = db.query(MCPServerConfig).filter(MCPServerConfig.is_active.is_(True))
    if api_key.id != "master":
        query = query.filter(MCPServerConfig.owner_id == api_key.id)

    servers = query.all()
    data = [MCPServerResponse.model_validate(s) for s in servers]
    return APIResponse(data=data)


@router.post("/", response_model=APIResponse[MCPServerResponse])
@limiter.limit("5/minute")
async def register_mcp_server(
    request: Request,
    body: MCPServerCreate,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[MCPServerResponse]:
    if api_key.id == "master":
        raise HTTPException(403, detail="Master key cannot register MCP servers directly.")

    # Check if server with same name already exists for this owner
    existing = (
        db.query(MCPServerConfig)
        .filter(
            MCPServerConfig.name == body.name,
            MCPServerConfig.owner_id == api_key.id,
            MCPServerConfig.is_active.is_(True),
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400, detail=f"MCP Server with name '{body.name}' is already registered."
        )

    server = MCPServerConfig(**body.model_dump(), owner_id=api_key.id)
    db.add(server)
    db.commit()
    db.refresh(server)
    return APIResponse(data=MCPServerResponse.model_validate(server))


@router.post("/{server_id}/test", response_model=APIResponse)
@limiter.limit("10/minute")
async def test_mcp_server(
    request: Request,
    server_id: str,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    """Attempt to connect and list available tools from the server."""
    query = db.query(MCPServerConfig).filter(MCPServerConfig.id == server_id)
    if api_key.id != "master":
        query = query.filter(MCPServerConfig.owner_id == api_key.id)

    server = query.first()
    if not server:
        raise HTTPException(404, "Server not found.")
    config: dict[str, object] = {
        "name": server.name,
        "transport": server.transport,
        "url": server.url,
        "command": server.command,
        "args": server.args,
        "env": server.env,
    }
    tools = await load_mcp_tools(config)
    return APIResponse(
        data={
            "server": server.name,
            "tools_found": len(tools),
            "tool_names": [t.name for t in tools],
        }
    )


@router.delete("/{server_id}", response_model=APIResponse)
@limiter.limit("5/minute")
async def deregister_mcp_server(
    request: Request,
    server_id: str,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    query = db.query(MCPServerConfig).filter(MCPServerConfig.id == server_id)
    if api_key.id != "master":
        query = query.filter(MCPServerConfig.owner_id == api_key.id)

    server = query.first()
    if not server:
        raise HTTPException(404, "Server not found.")
    server.is_active = False  # type: ignore[assignment]
    db.commit()
    return APIResponse(data={"message": f"Server '{server.name}' deregistered."})
