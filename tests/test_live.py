from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


def test_live_ws_handshake_missing_api_key(client: TestClient):
    """Verify WS connection closes when no api_key is provided in the handshake."""
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket.send_text(json.dumps({}))  # missing api_key
            websocket.receive_text()  # server should close with 1008


def test_live_ws_handshake_invalid_api_key(client: TestClient):
    """Verify WS connection closes with an invalid api_key."""
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket.send_text(json.dumps({"api_key": "invalid-key"}))
            websocket.receive_text()  # server should close with 1008


def test_live_ws_handshake_success(client: TestClient):
    """Verify WS connection succeeds after a valid handshake."""
    master_key = "test-master-key"
    with patch("app.routers.live.live_session_handler", new_callable=AsyncMock) as mock_handler:
        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket.send_text(json.dumps({"api_key": master_key}))
        mock_handler.assert_awaited_once()
