from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from functools import lru_cache
from typing import Any

from fastapi import HTTPException
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.agents import PRESETS, AgentConfig, build_agent, run_once
from app.database.models import Agents, APIKey, MCPServerConfig, Thread, ThreadMessage
from app.mcp.client import load_mcp_tools
from app.memory.checkpointer import get_checkpointer
from app.services.gemini import generate_thread_title, resolve_attachments

CompiledGraph = Any

logger = logging.getLogger(__name__)


class AgentCreate(BaseModel):
    name: str
    description: str | None = None
    system_prompt: str
    tools: list[str]
    model: str


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    system_prompt: str
    tools: list[str]
    model: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentRunRequest(BaseModel):
    model: str | None = None
    preset: str | None = None  # hardcoded preset name
    agent_id: str | None = None  # DB-backed config id — takes priority
    mcp_server_ids: list[str] | None = None  # external MCP servers to connect to
    prompt: str
    thread_id: str | None = None
    attachments: list[str] = []

    @model_validator(mode="after")
    def check_source(self) -> AgentRunRequest:
        if bool(self.preset) == bool(self.agent_id):
            raise ValueError("Exactly one of 'preset' or 'agent_id' is required.")
        if self.preset and not self.model:
            raise ValueError("'model' is required when using a 'preset'.")
        return self


class AgentRunResponse(BaseModel):
    answer: str
    preset: str | None = None
    model: str
    thread_id: str


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    model: str | None = None


@lru_cache(maxsize=32)
def _get_cached_agent(preset: str, model: str, checkpointer_id: int) -> CompiledGraph:
    """
    Cached agent factory to avoid rebuilding the agent on every request.
    Includes checkpointer_id in the cache key to ensure the checkpointer is correctly handled.
    """
    checkpointer: Any = get_checkpointer()
    agent_factory = PRESETS[preset]
    return agent_factory(model=model, checkpointer=checkpointer)


async def _get_agent(
    preset: str, model: str, checkpointer: Any, extra_tools: list[BaseTool] | None = None
) -> CompiledGraph:
    """Helper to get cached or new agent with optional extra tools."""
    if extra_tools:
        # Bypass cache if extra tools are present as they are not hashable
        agent_factory = PRESETS[preset]
        return await run_in_threadpool(
            agent_factory, model=model, checkpointer=checkpointer, extra_tools=extra_tools
        )

    return await run_in_threadpool(_get_cached_agent, preset, model, id(checkpointer))


async def run_agent_service(
    request: AgentRunRequest, db: Session, api_key: APIKey
) -> AgentRunResponse:
    """
    Unified service for running agents.
    """
    system_prompt: str = ""
    tools: list[str] = []
    model: str = ""
    preset_name: str = ""

    # Determine agent config
    if request.agent_id:
        agent_query = db.query(Agents).filter(
            Agents.id == request.agent_id,
            Agents.is_active.is_(True),
        )
        if api_key.id != "master":
            agent_query = agent_query.filter(Agents.owner_id == api_key.id)

        agent_config_model = agent_query.first()
        if not agent_config_model:
            raise HTTPException(
                status_code=404, detail=f"Agent config {request.agent_id} not found"
            )

        system_prompt = str(agent_config_model.system_prompt)
        tools = list(agent_config_model.tools) if agent_config_model.tools else []
        model = str(agent_config_model.model)
        preset_name = f"custom:{agent_config_model.name}"
    else:
        preset = str(request.preset)
        if preset not in PRESETS:
            raise HTTPException(
                status_code=400, detail=f"Invalid preset. Available: {list(PRESETS.keys())}"
            )

        model = str(request.model)
        preset_name = preset

    # Handle threading
    thread: Thread | None = None
    if request.thread_id:
        thread_query = db.query(Thread).filter(Thread.id == request.thread_id)
        if api_key.id != "master":
            thread_query = thread_query.filter(Thread.owner_id == api_key.id)

        thread = thread_query.first()
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread {request.thread_id} not found")
        if thread.preset != preset_name or thread.model != model:
            raise HTTPException(
                status_code=400, detail="Thread belongs to a different agent configuration."
            )
    else:
        # Generate title for new thread
        title = await run_in_threadpool(generate_thread_title, request.prompt, model)
        thread = Thread(
            id=str(uuid.uuid4()),
            owner_id=api_key.id,
            preset=preset_name,
            model=model,
            title=title,
        )
        db.add(thread)
        db.commit()
        db.refresh(thread)

    try:
        logger.info(
            f"Running agent with preset: {preset_name}, model: {model}, thread: {thread.id}"
        )
        checkpointer = get_checkpointer()

        # Load additional MCP tools if requested
        mcp_tools: list[BaseTool] = []
        if request.mcp_server_ids:
            for server_id in request.mcp_server_ids:
                mcp_query = db.query(MCPServerConfig).filter(
                    MCPServerConfig.id == server_id, MCPServerConfig.is_active.is_(True)
                )
                if api_key.id != "master":
                    mcp_query = mcp_query.filter(MCPServerConfig.owner_id == api_key.id)

                server_config = mcp_query.first()
                if server_config:
                    config_dict: dict[str, object] = {
                        "name": server_config.name,
                        "transport": server_config.transport,
                        "url": server_config.url,
                        "command": server_config.command,
                        "args": server_config.args,
                        "env": server_config.env,
                    }
                    tools_from_server = await load_mcp_tools(config_dict)
                    mcp_tools.extend(tools_from_server)

        # Build or get cached agent
        agent: CompiledGraph
        if request.agent_id:
            # Merge agent's native tools with MCP tools
            combined_tools: list[str | BaseTool] = list(tools) + list(mcp_tools)
            agent = await run_in_threadpool(
                build_agent,
                tools=combined_tools,
                system_prompt=system_prompt,
                model=model,
                checkpointer=checkpointer,
            )
        else:
            agent = await _get_agent(preset_name, model, checkpointer, extra_tools=mcp_tools)

        # Run the agent
        config = AgentConfig(
            name=f"{preset_name.capitalize()} Agent",
            model=model,
            verbose=False,
        )

        lg_config: dict[str, Any] = {"configurable": {"thread_id": thread.id}}
        prompt_input: Any = request.prompt
        if request.attachments:
            resolved = resolve_attachments(request.attachments, db, str(api_key.id))
            prompt_input = [{"type": "text", "text": request.prompt}]
            for att in resolved:
                prompt_input.append(
                    {"type": "media", "file_uri": att["uri"], "mime_type": att["mime_type"]}
                )

        answer = await run_in_threadpool(
            run_once, agent, prompt_input, config=config, lg_config=lg_config
        )

        # Save messages
        human_msg = ThreadMessage(
            id=str(uuid.uuid4()), thread_id=thread.id, role="human", content=request.prompt
        )
        db.add(human_msg)
        ai_msg = ThreadMessage(id=str(uuid.uuid4()), thread_id=thread.id, role="ai", content=answer)
        db.add(ai_msg)
        db.commit()

        return AgentRunResponse(
            answer=answer, preset=preset_name, model=model, thread_id=str(thread.id)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error running agent")
        raise HTTPException(status_code=500, detail="Agent execution failed.") from e


async def run_agent_stream_service(
    request: AgentRunRequest, db: Session, api_key: APIKey
) -> AsyncGenerator[str, None]:
    """
    Unified service for running agents with streaming responses.
    """
    system_prompt: str = ""
    tools: list[str] = []
    model: str = ""
    preset_name: str = ""

    # Determine agent config
    if request.agent_id:
        agent_query = db.query(Agents).filter(
            Agents.id == request.agent_id, Agents.is_active.is_(True)
        )
        if api_key.id != "master":
            agent_query = agent_query.filter(Agents.owner_id == api_key.id)

        agent_config_model = agent_query.first()
        if not agent_config_model:
            raise HTTPException(
                status_code=404, detail=f"Agent config {request.agent_id} not found"
            )

        system_prompt = str(agent_config_model.system_prompt)
        tools = list(agent_config_model.tools) if agent_config_model.tools else []
        model = str(agent_config_model.model)
        preset_name = f"custom:{agent_config_model.name}"
    else:
        preset = str(request.preset)
        if preset not in PRESETS:
            raise HTTPException(
                status_code=400, detail=f"Invalid preset. Available: {list(PRESETS.keys())}"
            )

        model = str(request.model)
        preset_name = preset

    # Handle threading
    thread: Thread | None = None
    if request.thread_id:
        thread_query = db.query(Thread).filter(Thread.id == request.thread_id)
        if api_key.id != "master":
            thread_query = thread_query.filter(Thread.owner_id == api_key.id)

        thread = thread_query.first()
        if not thread:
            raise HTTPException(status_code=404, detail=f"Thread {request.thread_id} not found")

        # Verify thread configuration matches incoming run
        if thread.preset != preset_name or thread.model != model:
            raise HTTPException(
                status_code=400, detail="Thread belongs to a different agent configuration."
            )
    else:
        # Generate title for new thread
        title = await run_in_threadpool(generate_thread_title, request.prompt, model)
        thread = Thread(
            id=str(uuid.uuid4()),
            owner_id=api_key.id,
            preset=preset_name,
            model=model,
            title=title,
        )
        db.add(thread)
        db.commit()
        db.refresh(thread)

    # Save Human message
    human_msg = ThreadMessage(
        id=str(uuid.uuid4()), thread_id=thread.id, role="human", content=request.prompt
    )
    db.add(human_msg)
    db.commit()

    try:
        logger.info(
            f"Streaming agent with preset: {preset_name}, model: {model}, thread: {thread.id}"
        )
        checkpointer = get_checkpointer()

        # Load additional MCP tools if requested
        mcp_tools: list[BaseTool] = []
        if request.mcp_server_ids:
            for server_id in request.mcp_server_ids:
                mcp_query = db.query(MCPServerConfig).filter(
                    MCPServerConfig.id == server_id, MCPServerConfig.is_active.is_(True)
                )
                if api_key.id != "master":
                    mcp_query = mcp_query.filter(MCPServerConfig.owner_id == api_key.id)

                server_config = mcp_query.first()
                if server_config:
                    config_dict: dict[str, object] = {
                        "name": server_config.name,
                        "transport": server_config.transport,
                        "url": server_config.url,
                        "command": server_config.command,
                        "args": server_config.args,
                        "env": server_config.env,
                    }
                    tools_from_server = await load_mcp_tools(config_dict)
                    mcp_tools.extend(tools_from_server)

        # Build or get cached agent
        agent: CompiledGraph
        if request.agent_id:
            combined_tools: list[str | BaseTool] = list(tools) + list(mcp_tools)
            agent = await run_in_threadpool(
                build_agent,
                tools=combined_tools,
                system_prompt=system_prompt,
                model=model,
                checkpointer=checkpointer,
            )
        else:
            agent = await _get_agent(preset_name, model, checkpointer, extra_tools=mcp_tools)

        lg_config: dict[str, Any] = {"configurable": {"thread_id": thread.id}}
        prompt_input: Any = request.prompt
        if request.attachments:
            resolved = resolve_attachments(request.attachments, db, str(api_key.id))
            prompt_input = [{"type": "text", "text": request.prompt}]
            for att in resolved:
                prompt_input.append(
                    {"type": "media", "file_uri": att["uri"], "mime_type": att["mime_type"]}
                )

        full_answer = ""

        async for event in agent.astream_events(
            {"messages": [("human", prompt_input)]}, config=lg_config, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    if isinstance(chunk, str):
                        full_answer += chunk
                    elif isinstance(chunk, list):
                        for part in chunk:
                            if isinstance(part, dict) and part.get("type") == "text":
                                full_answer += str(part.get("text", ""))
                            elif isinstance(part, str):
                                full_answer += part
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name'], 'input': event['data'].get('input')}, default=str)}\n\n"
            elif kind == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output': event['data'].get('output')}, default=str)}\n\n"

        # Save AI message
        ai_msg = ThreadMessage(
            id=str(uuid.uuid4()), thread_id=thread.id, role="ai", content=full_answer
        )
        db.add(ai_msg)
        db.commit()

        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread.id})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("Error in streaming agent")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Stream failed'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread.id})}\n\n"
        yield "data: [DONE]\n\n"
