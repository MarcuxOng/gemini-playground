from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database.db import Base


class APIKey(Base):
    __tablename__ = "playground_v1_api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String, index=True)
    hashed_key = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    revoked_at = Column(DateTime, nullable=True)


class Thread(Base):
    __tablename__ = "playground_v1_threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, ForeignKey("playground_v1_api_keys.id"), nullable=False, index=True)
    title = Column(String, nullable=True)
    preset = Column(String, nullable=False)
    model = Column(String, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    messages = relationship(
        "ThreadMessage",
        back_populates="thread",
        order_by="ThreadMessage.created_at",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ThreadMessage(Base):
    __tablename__ = "playground_v1_thread_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(
        String, ForeignKey("playground_v1_threads.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String, nullable=False)  # "human" | "ai" | "tool"
    content = Column(Text, nullable=False)
    tool_calls = Column(JSON, nullable=True)  # raw tool call data if role=="ai"
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    thread = relationship("Thread", back_populates="messages")


class Agents(Base):
    __tablename__ = "playground_v1_agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, ForeignKey("playground_v1_api_keys.id"), nullable=False, index=True)
    name = Column(
        String, nullable=False
    )  # Removed unique=True to allow same slug for different users
    description = Column(String, nullable=True)
    system_prompt = Column(Text, nullable=False)
    tools = Column(JSON, nullable=False, default=list)  # list of tool name strings
    model = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_agents_owner_name"),)


class MCPServerConfig(Base):
    """
    Configuration for an external MCP server your agents can connect to.
    """

    __tablename__ = "playground_v1_mcp_servers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String, ForeignKey("playground_v1_api_keys.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    transport = Column(String, nullable=True)  # "sse" | "stdio" (can be inferred)
    url = Column(String, nullable=True)  # for SSE transport
    command = Column(String, nullable=True)  # for stdio: e.g. "npx"
    args = Column(JSON, nullable=True)  # ["@modelcontextprotocol/server-filesystem"]
    env = Column(JSON, nullable=True)  # env vars for stdio servers
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class EvalDataset(Base):
    __tablename__ = "playground_v1_eval_datasets"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    cases = Column(JSON, nullable=False)  # [{input, expected, ...}]
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class EvalRun(Base):
    __tablename__ = "playground_v1_eval_runs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("playground_v1_eval_datasets.id"))
    agent_id = Column(String, nullable=False)  # preset name or custom agent UUID
    metrics = Column(JSON, nullable=False)  # {passed: int, failed: int, total: int, results: [...]}
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
