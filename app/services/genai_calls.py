"""
Raw-client calls that need the regional -> global endpoint fallback.

Some models are served from only one of the two Vertex endpoints (e.g.
gemini-3.5-flash 404s on us-central1 but works on global; gemini-embedding-2 is
global-only). Every raw `genai` generate call goes through here so the retry
behaves identically for text, RAG, and image paths — this used to be copied
verbatim into app/services/image.py.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from google.genai import errors, types

from app.config import client, global_client

logger = logging.getLogger(__name__)


def generate_content_with_fallback(
    model: str, contents: Any, config: types.GenerateContentConfig
) -> types.GenerateContentResponse:
    """Call the regional endpoint, retrying on global if the model 404s there."""
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except errors.ClientError as e:
        if e.code != 404:
            raise
        logger.warning(
            f"Model {model!r} not found on regional endpoint, retrying on global: {e.message}"
        )
        return global_client.models.generate_content(model=model, contents=contents, config=config)


async def generate_content_stream_with_fallback(
    model: str, contents: Any, config: types.GenerateContentConfig
) -> AsyncIterator[types.GenerateContentResponse]:
    """Streaming counterpart to generate_content_with_fallback."""
    try:
        return await client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
    except errors.ClientError as e:
        if e.code != 404:
            raise
        logger.warning(
            f"Model {model!r} not found on regional endpoint, retrying on global: {e.message}"
        )
        return await global_client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
