from __future__ import annotations

import hashlib
import logging
import secrets

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import Settings, get_settings, settings
from app.database.db import get_db
from app.database.models import APIKey

logger = logging.getLogger(__name__)


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256 for secure DB lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def check_api_key(api_key: str, db: Session) -> bool:
    """
    Synchronous check for API key validity.
    Checks against both the master key and the database.
    """
    # 1. Check Master Key
    if settings.master_api_key and secrets.compare_digest(api_key, settings.master_api_key):
        return True

    # 2. Check Database Keys
    hashed = hash_api_key(api_key)
    api_key_record = (
        db.query(APIKey).filter(APIKey.hashed_key == hashed, APIKey.is_active.is_(True)).first()
    )

    return api_key_record is not None


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> APIKey:
    """
    Dependency to verify API keys by hashing and checking the DB.
    Returns the APIKey record for the authenticated user.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    # 1. Check Master Key
    if settings.master_api_key and secrets.compare_digest(x_api_key, settings.master_api_key):
        request.state.api_key_id = "master"
        return APIKey(id="master", name="Master Key")

    # 2. Check Database Keys
    hashed = hash_api_key(x_api_key)
    api_key_record = (
        db.query(APIKey).filter(APIKey.hashed_key == hashed, APIKey.is_active.is_(True)).first()
    )

    if not api_key_record:
        logger.warning("Unauthorized API access attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    request.state.api_key_id = api_key_record.id
    return api_key_record


async def verify_master_key(
    request: Request, x_api_key: str = Header(...), settings: Settings = Depends(get_settings)
) -> None:
    """
    Dependency that only allows requests using the MASTER_API_KEY.
    Used for administrative endpoints like creating/listing keys.
    """
    if not settings.master_api_key or not secrets.compare_digest(
        x_api_key, settings.master_api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Administrative privileges required.",
        )

    request.state.api_key_id = "master"


async def verify_internal_key(
    request: Request,
    x_internal_key: str = Header(..., alias="x-internal-key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Dependency for server-to-server internal endpoints.
    Uses a dedicated internal_api_key separate from user API keys.
    """
    if not settings.internal_api_key or not secrets.compare_digest(
        x_internal_key, settings.internal_api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Internal access only.",
        )
