from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agents import PRESETS
from app.database.db import get_db
from app.database.models import Agents, APIKey
from app.services.agents import (
    AgentCreate,
    AgentResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentUpdate,
    run_agent_service,
    run_agent_stream_service,
)
from app.tools import get_registry, has_tool, list_tool_names
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Agents"],
)


@router.get("/list", response_model=APIResponse[list[AgentResponse]])
async def list_agents(
    db: Session = Depends(get_db), api_key: APIKey = Depends(verify_api_key)
) -> APIResponse[list[AgentResponse]]:
    query = db.query(Agents).filter(Agents.is_active.is_(True))
    if api_key.id != "master":
        query = query.filter(Agents.owner_id == api_key.id)

    configs = query.all()
    return APIResponse(data=[AgentResponse.model_validate(c) for c in configs])


@router.get("/presets", response_model=APIResponse)
async def get_presets(api_key: APIKey = Depends(verify_api_key)) -> APIResponse:  # type: ignore[type-arg]
    """List available agent presets."""
    return APIResponse(data={"presets": list(PRESETS.keys())})


@router.get("/tools", response_model=APIResponse)
async def list_available_tools(api_key: APIKey = Depends(verify_api_key)) -> APIResponse:  # type: ignore[type-arg]
    """List all registered tools that can be assigned to an agent config."""
    tools = [
        {
            "name": name,
            "description": entry["schema"]["function"]["description"],
            "parameters": entry["schema"]["function"]["parameters"],
        }
        for name, entry in get_registry().items()
    ]
    return APIResponse(data=tools)


@router.post("/create", response_model=APIResponse[AgentResponse])
@limiter.limit("10/minute")
async def create_agent(
    request: Request,
    body: AgentCreate,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[AgentResponse]:
    if api_key.id == "master":
        raise HTTPException(403, detail="Master key cannot create agents directly.")

    # Validate requested tools exist in registry
    unknown = [t for t in body.tools if not has_tool(t)]
    if unknown:
        raise HTTPException(400, detail=f"Unknown tools: {unknown}, Available: {list_tool_names()}")

    config = Agents(**body.model_dump(), owner_id=api_key.id)
    db.add(config)
    db.commit()
    db.refresh(config)
    return APIResponse(data=AgentResponse.model_validate(config))


@router.patch("/{agent_id}", response_model=APIResponse[AgentResponse])
@limiter.limit("10/minute")
async def update_agent_config(
    request: Request,
    agent_id: str,
    body: AgentUpdate,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[AgentResponse]:
    query = db.query(Agents).filter(Agents.id == agent_id)
    if api_key.id != "master":
        query = query.filter(Agents.owner_id == api_key.id)

    config = query.first()

    if not config:
        raise HTTPException(404, "Config not found.")

    patch = body.model_dump(exclude_unset=True, exclude_none=True)
    if "tools" in patch:
        unknown = [t for t in patch["tools"] if not has_tool(t)]
        if unknown:
            raise HTTPException(
                400, detail=f"Unknown tools: {unknown}, Available: {list_tool_names()}"
            )

    for field, value in patch.items():
        setattr(config, field, value)
    db.commit()
    db.refresh(config)
    return APIResponse(data=AgentResponse.model_validate(config))


@router.delete("/{agent_id}", response_model=APIResponse)
@limiter.limit("10/minute")
async def delete_agent_config(
    request: Request,
    agent_id: str,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    query = db.query(Agents).filter(Agents.id == agent_id)
    if api_key.id != "master":
        query = query.filter(Agents.owner_id == api_key.id)

    config = query.first()

    if not config:
        raise HTTPException(404, "Config not found.")
    config.is_active = False  # type: ignore[assignment]
    db.commit()
    return APIResponse(data={"message": f"Config {agent_id} deactivated."})


@router.post("/run", response_model=APIResponse[AgentRunResponse])
@limiter.limit("20/minute")
async def run_agent(
    request: Request,
    body: AgentRunRequest,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[AgentRunResponse]:
    """
    Unified endpoint for running agents.
    """
    logger.info(f"Calling agents API with model: {body.model}")
    response = await run_agent_service(body, db, api_key)
    return APIResponse(data=response)


@router.post("/run/stream")
@limiter.limit("20/minute")
async def run_agent_stream(
    request: Request,
    body: AgentRunRequest,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> StreamingResponse:
    """
    Endpoint for running agents with streaming responses.
    """
    logger.info(f"Starting agent stream with model: {body.model}")
    response = StreamingResponse(
        run_agent_stream_service(body, db, api_key), media_type="text/event-stream"
    )
    return response
