from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from app.config import build_genai_client

logger = logging.getLogger(__name__)
client = build_genai_client()


def _dict_from_cache(
    cache: types.CachedContent, fallback_id: str = "", fallback_ttl: str = ""
) -> dict[str, Any]:
    """Serialize a CachedContent object to a consistent dict representation."""
    return {
        "cache_id": str(cache.name or fallback_id),
        "model": str(cache.model or ""),
        "display_name": str(cache.display_name or ""),
        "ttl": str(_ttl) if (_ttl := getattr(cache, "ttl", None)) else fallback_ttl,
        "create_time": str(cache.create_time) if cache.create_time else None,
        "expire_time": str(cache.expire_time) if cache.expire_time else None,
    }


def create_context_cache(
    model: str,
    file_uris: list[str],
    mime_types: list[str],
    system_instruction: str | None = None,
    display_name: str | None = None,
    ttl: str = "3600s",
) -> dict[str, Any]:
    """Create a Gemini context cache with file URIs and optional system instruction."""
    contents: list[types.Part] = []
    for uri, mime in zip(file_uris, mime_types, strict=True):
        contents.append(types.Part.from_uri(file_uri=uri, mime_type=mime))

    config = types.CreateCachedContentConfig(
        contents=contents if contents else None,
        system_instruction=system_instruction,
        ttl=ttl,
        display_name=display_name or "context-cache",
    )

    cache = client.caches.create(model=model, config=config)
    return _dict_from_cache(cache, fallback_ttl=ttl)


def get_cache(cache_id: str) -> dict[str, Any]:
    """Get a context cache by ID."""
    cache = client.caches.get(name=cache_id)
    return _dict_from_cache(cache, fallback_id=cache_id)


def list_caches() -> list[dict[str, Any]]:
    """List all context caches."""
    return [_dict_from_cache(c) for c in client.caches.list()]


def delete_cache(cache_id: str) -> None:
    """Delete a context cache."""
    client.caches.delete(name=cache_id)


def update_cache_ttl(cache_id: str, ttl: str) -> dict[str, Any]:
    """Update the TTL of a context cache."""
    config = types.UpdateCachedContentConfig(ttl=ttl)
    cache = client.caches.update(name=cache_id, config=config)
    return _dict_from_cache(cache, fallback_id=cache_id, fallback_ttl=ttl)
