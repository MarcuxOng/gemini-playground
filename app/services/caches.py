from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from app.config import build_genai_client

logger = logging.getLogger(__name__)
client = build_genai_client()


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

    return {
        "cache_id": str(cache.name or ""),
        "model": str(cache.model or model),
        "display_name": str(cache.display_name or ""),
        "ttl": str(cache.ttl) if cache.ttl else ttl,
        "create_time": str(cache.create_time) if cache.create_time else None,
        "expire_time": str(cache.expire_time) if cache.expire_time else None,
    }


def get_cache(cache_id: str) -> dict[str, Any]:
    """Get a context cache by ID."""
    cache = client.caches.get(name=cache_id)
    return {
        "cache_id": str(cache.name or cache_id),
        "model": str(cache.model or ""),
        "display_name": str(cache.display_name or ""),
        "ttl": str(cache.ttl) if cache.ttl else None,
        "create_time": str(cache.create_time) if cache.create_time else None,
        "expire_time": str(cache.expire_time) if cache.expire_time else None,
    }


def list_caches() -> list[dict[str, Any]]:
    """List all context caches."""
    caches_list = client.caches.list()
    results: list[dict[str, Any]] = []
    for cache in caches_list:
        results.append(
            {
                "cache_id": str(cache.name or ""),
                "model": str(cache.model or ""),
                "display_name": str(cache.display_name or ""),
                "ttl": str(cache.ttl) if cache.ttl else None,
                "create_time": str(cache.create_time) if cache.create_time else None,
                "expire_time": str(cache.expire_time) if cache.expire_time else None,
            }
        )
    return results


def delete_cache(cache_id: str) -> None:
    """Delete a context cache."""
    client.caches.delete(name=cache_id)


def update_cache_ttl(cache_id: str, ttl: str) -> dict[str, Any]:
    """Update the TTL of a context cache."""
    config = types.UpdateCachedContentConfig(ttl=ttl)
    cache = client.caches.update(name=cache_id, config=config)
    return {
        "cache_id": str(cache.name or cache_id),
        "model": str(cache.model or ""),
        "display_name": str(cache.display_name or ""),
        "ttl": str(cache.ttl) if cache.ttl else ttl,
        "create_time": str(cache.create_time) if cache.create_time else None,
        "expire_time": str(cache.expire_time) if cache.expire_time else None,
    }
