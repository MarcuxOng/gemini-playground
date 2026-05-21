from __future__ import annotations

import json
import logging
import secrets

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketState

from app.config import Settings, get_settings
from app.database.db import get_db
from app.database.models import APIKey
from app.services.live import live_session_handler
from app.utils.auth import hash_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/live", tags=["Live"])


def verify_ws_api_key(db: Session, settings: Settings, api_key: str | None = None) -> APIKey:
    """Manual API key verification for WebSockets."""
    if not api_key:
        raise ValueError("API key required")

    # 1. Check Master Key
    if settings.master_api_key and secrets.compare_digest(api_key, settings.master_api_key):
        return APIKey(id="master", name="Master Key")

    # 2. Check Database Keys
    hashed = hash_api_key(api_key)
    key_rec = (
        db.query(APIKey).filter(APIKey.hashed_key == hashed, APIKey.is_active.is_(True)).first()
    )
    if not key_rec:
        raise ValueError("Invalid or inactive API key")

    return key_rec


@router.websocket("/ws")
async def live_ws_endpoint(
    websocket: WebSocket,
    model: str = Query("gemini-2.0-flash-exp"),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    WebSocket endpoint for Gemini Live API.
    Authenticates via an initial JSON handshake message: {"api_key": "<key>"}.
    The optional 'model' query parameter selects the Gemini model.
    """
    await websocket.accept()
    try:
        # 1. Auth handshake — client must send {"api_key": "..."} as first message
        raw = await websocket.receive_text()
        try:
            auth_data = json.loads(raw)
            api_key = auth_data.get("api_key")
        except (json.JSONDecodeError, AttributeError):
            await websocket.close(code=1008, reason="Invalid auth handshake")
            return

        try:
            verify_ws_api_key(db, settings, api_key)
        except ValueError as e:
            await websocket.close(code=1008, reason=str(e))
            return

        logger.info(f"Accepted Live WS connection. Model: {model}")

        # 2. Hand off to service handler
        await live_session_handler(websocket, model=model)

    except WebSocketDisconnect:
        logger.info("Live WS connection closed by client.")
    except Exception as e:
        logger.error(f"Error in Live WS endpoint: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason="Internal server error")
