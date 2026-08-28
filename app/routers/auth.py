from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import APIKey
from app.utils.auth import hash_api_key, verify_master_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class APIKeyResponse(BaseModel):
    id: str
    name: str
    is_active: bool
    created_at: datetime
    revoked_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


@router.post("/keys/generate", response_model=APIResponse)
@limiter.limit("5/minute")
async def generate_key(
    request: Request, name: str, db: Session = Depends(get_db), _: None = Depends(verify_master_key)
) -> APIResponse:  # type: ignore[type-arg]
    """Generate a new, unique API key."""
    # Create raw key (e.g. "sk_play_...")
    raw_key = f"sk_play_{secrets.token_urlsafe(32)}"
    hashed_key = hash_api_key(raw_key)

    new_key = APIKey(name=name, hashed_key=hashed_key, is_active=True)

    db.add(new_key)
    db.commit()
    db.refresh(new_key)

    logger.info(f"Generated new API key: {name}")
    return APIResponse(
        data={
            "api_key": raw_key,
            "name": name,
            "note": "Save this key now — it cannot be recovered later.",
        }
    )


@router.get("/keys", response_model=APIResponse[list[APIKeyResponse]])
@limiter.limit("20/minute")
async def list_keys(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_master_key),
) -> APIResponse[list[APIKeyResponse]]:
    """List all registered API keys."""
    keys = db.query(APIKey).all()
    return APIResponse(data=[APIKeyResponse.model_validate(k) for k in keys])


@router.delete("/keys/{key_id}", response_model=APIResponse)
@limiter.limit("10/minute")
async def revoke_key(
    request: Request,
    key_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_master_key),
) -> APIResponse:  # type: ignore[type-arg]
    """Revoke (deactivate) an existing API key."""
    api_key_record = db.query(APIKey).filter(APIKey.id == key_id).first()

    if not api_key_record:
        raise HTTPException(status_code=404, detail="API Key not found.")

    api_key_record.is_active = False  # type: ignore[assignment]
    api_key_record.revoked_at = datetime.now(UTC)  # type: ignore[assignment]
    db.commit()

    logger.info(f"Revoked API key ID: {key_id}")
    return APIResponse(data={"message": f"Successfully revoked key: {api_key_record.name}"})
