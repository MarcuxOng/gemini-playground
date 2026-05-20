from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_live_ws_handshake_missing_api_key():
    """Verify WS connection fails without API key."""
    with pytest.raises(Exception): # TestClient raises if handshake fails
        with client.websocket_connect("/api/v1/live/ws"):
            pass

def test_live_ws_handshake_invalid_api_key(client: TestClient):
    """Verify WS connection fails with invalid API key."""
    # Note: TestClient.websocket_connect will raise if server closes with non-1000/1001
    # We expect 1008 (Policy Violation) for auth failure
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/live/ws?api_key=invalid"):
            pass

def test_live_ws_handshake_success(client: TestClient):
    """Verify WS connection succeeds with valid API key."""
    # The master key is "test-master-key" in conftest.py
    master_key = "test-master-key"
    # We need to mock the live_session_handler to avoid real Gemini connection
    from unittest.mock import patch
    with patch("app.routers.live.live_session_handler") as mock_handler:
        mock_handler.return_value = None
        with client.websocket_connect(f"/api/v1/live/ws?api_key={master_key}") as websocket:
            # If we reach here, handshake succeeded
            assert True
