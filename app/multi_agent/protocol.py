"""Multimodal Inter-Agent Protocol (MIAP).

Agents pass raw :class:`google.genai.types.Part` objects (image frames,
audio clips, PDFs) directly to each other — no lossy text transcription.

Endpoints that consume MIAP messages use ``x-internal-key`` auth (see
:func:`app.utils.auth.verify_internal_key`).
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentPart(BaseModel):
    """A multimodal part in an inter-agent message.

    Mirrors the three shapes of ``google.genai.types.Part`` that are
    useful for agent-to-agent communication:

    * ``text`` — plain text content.
    * ``inline_data`` — base64-encoded binary payload (image, audio, etc.).
    * ``file_uri`` — reference to a previously uploaded Gemini file.

    Only one payload field should be set per part, matching the declared
    ``type``.
    """

    type: Literal["text", "inline_data", "file_uri"]
    text: str | None = None
    mime_type: str | None = None
    data: str | None = None
    file_uri: str | None = None


class AgentMessage(BaseModel):
    """Message passed between agents via the Multimodal Inter-Agent Protocol.

    Sender agents construct a message containing raw ``AgentPart`` items
    and POST it to the target agent's ``/api/v1/agents/invoke`` endpoint.
    """

    parts: list[AgentPart] = Field(..., min_length=1)
    sender_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def agent_part_to_gemini_part(part: AgentPart) -> types.Part:
    """Convert an ``AgentPart`` to the native ``google.genai.types.Part``.

    Raises :class:`ValueError` if the part payload does not match its
    declared ``type``.
    """
    if part.type == "text":
        if not part.text:
            raise ValueError("AgentPart type='text' requires 'text' field to be set.")
        return types.Part.from_text(text=part.text)

    if part.type == "inline_data":
        if not part.data:
            raise ValueError("AgentPart type='inline_data' requires 'data' field to be set.")
        if not part.mime_type:
            raise ValueError("AgentPart type='inline_data' requires 'mime_type' field to be set.")
        decoded = base64.b64decode(part.data)
        return types.Part.from_bytes(data=decoded, mime_type=part.mime_type)

    if part.type == "file_uri":
        if not part.file_uri:
            raise ValueError("AgentPart type='file_uri' requires 'file_uri' field to be set.")
        return types.Part.from_uri(file_uri=part.file_uri, mime_type=part.mime_type or "")

    raise ValueError(f"Unknown AgentPart type: {part.type}")


def agent_message_to_gemini_parts(message: AgentMessage) -> list[types.Part]:
    """Convert every part in an ``AgentMessage`` to native Gemini Parts."""
    return [agent_part_to_gemini_part(p) for p in message.parts]
