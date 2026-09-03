"""Auth — the API key lifecycle.

What this replaces asserted nothing. It asked for `/api/v1/auth/login` and
`/api/v1/auth/verify` — neither route exists — and accepted `404`, so both tests
passed *because* the endpoints were absent. The router is key management:
generate, list, revoke, all master-only.

Note on the rate limit: `generate` is capped at 5/minute and the limiter is shared
across the session, but slowapi's decorator wraps the endpoint body, which never runs
when `verify_master_key` rejects first — so only successful creates count against it.
These tests make two.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

MASTER_ONLY_ROUTES = [
    ("post", "/api/v1/auth/keys/generate?name=probe"),
    ("get", "/api/v1/auth/keys"),
    ("delete", "/api/v1/auth/keys/some-id"),
]


@pytest.mark.parametrize(("method", "path"), MASTER_ONLY_ROUTES)
def test_key_management_rejects_a_non_master_key(client: TestClient, method: str, path: str):
    """`verify_master_key` is an allowlist of one — a valid non-master key is still 403."""
    response = getattr(client, method)(path, headers={"X-API-Key": "not-the-master-key"})

    assert response.status_code == 403


@pytest.mark.parametrize(("method", "path"), MASTER_ONLY_ROUTES)
def test_key_management_requires_the_header(client: TestClient, method: str, path: str):
    """The header is declared `Header(...)`, so its absence is a validation error, not a 403."""
    response = getattr(client, method)(path)

    assert response.status_code == 422


def test_generated_key_authenticates_and_is_returned_only_once(
    client: TestClient, auth_headers
):
    response = client.post("/api/v1/auth/keys/generate?name=probe-key", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["api_key"].startswith("sk_play_")
    # Only the hash is persisted, so the response is the one chance to keep it.
    assert "cannot be recovered" in data["note"]

    # The key is real: it authenticates against a route it did not come from.
    assert client.get("/api/v1/threads/", headers={"X-API-Key": data["api_key"]}).status_code == 200


def test_revoked_key_stops_authenticating(client: TestClient, auth_headers):
    """`verify_api_key` filters on `is_active`, so revocation has to take effect at once."""
    # `tests/test.db` is a real file that survives between runs, so a fixed name would
    # match a row left by an earlier run and this test would revoke the wrong key.
    name = f"doomed-key-{uuid.uuid4()}"
    created = client.post(
        f"/api/v1/auth/keys/generate?name={name}", headers=auth_headers
    ).json()["data"]
    raw_key = created["api_key"]

    listed = client.get("/api/v1/auth/keys", headers=auth_headers).json()["data"]
    key_id = next(k["id"] for k in listed if k["name"] == name)

    assert client.get("/api/v1/threads/", headers={"X-API-Key": raw_key}).status_code == 200

    revoked = client.delete(f"/api/v1/auth/keys/{key_id}", headers=auth_headers)
    assert revoked.status_code == 200

    assert client.get("/api/v1/threads/", headers={"X-API-Key": raw_key}).status_code == 401


def test_revoking_an_unknown_key_is_404(client: TestClient, auth_headers):
    response = client.delete("/api/v1/auth/keys/no-such-key-id", headers=auth_headers)

    assert response.status_code == 404
