from __future__ import annotations

import logging
from typing import Any

from google.genai import types
from sqlalchemy.orm import Session

from app.config import client
from app.database.models import CacheRecord

logger = logging.getLogger(__name__)


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


# ── Ownership ─────────────────────────────────────────────────────────────────
#
# Gemini context caches are project-scoped resources with no notion of this
# platform's API keys, so ownership is tracked here instead. Every read, reuse
# and delete path must go through record_cache_owner / assert_cache_access.


def record_cache_owner(
    db: Session, cache_id: str, owner_id: str, model: str, display_name: str | None
) -> None:
    """Persist the ownership row for a freshly created context cache."""
    # merge, not add: an upstream cache_id that already has a row would
    # otherwise raise IntegrityError and fail an otherwise successful create.
    db.merge(CacheRecord(id=cache_id, owner_id=owner_id, model=model, display_name=display_name))
    db.commit()


def owned_cache_ids(db: Session, owner_id: str) -> set[str]:
    """Return the cache IDs owned by *owner_id* ('master' sees every record)."""
    query = db.query(CacheRecord.id)
    if owner_id != "master":
        query = query.filter(CacheRecord.owner_id == owner_id)
    return {row[0] for row in query.all()}


def assert_cache_access(db: Session, cache_id: str, owner_id: str) -> None:
    """Raise PermissionError unless *owner_id* may use *cache_id*.

    A cache holds the contents of the documents it was built from, so an
    unowned cache_id is refused rather than silently ignored.
    """
    if owner_id == "master":
        return
    record = (
        db.query(CacheRecord)
        .filter(CacheRecord.id == cache_id, CacheRecord.owner_id == owner_id)
        .first()
    )
    if record is None:
        logger.warning("Denied cache access: %r requested by %r", cache_id, owner_id)
        raise PermissionError(f"Context cache {cache_id} not found or not accessible")


def delete_cache_record(db: Session, cache_id: str) -> None:
    """Drop the ownership row for a deleted cache."""
    db.query(CacheRecord).filter(CacheRecord.id == cache_id).delete()
    db.commit()
