"""Context caches are project-scoped Gemini resources with no owner of their own.

Before ``playground_v1_caches`` existed, every authenticated caller could list,
read, re-use and delete every other caller's cache — and a cache holds the
*contents* of the documents it was built from. These tests pin that shut.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.database.db import SessionLocal
from app.database.models import APIKey, CacheRecord
from app.utils.auth import hash_api_key

ALICE_KEY = "alice-plaintext-key"
BOB_KEY = "bob-plaintext-key"


@pytest.fixture(scope="module", autouse=True)
def two_users():
    """Two ordinary (non-master) API keys — the master key bypasses every filter."""
    db = SessionLocal()
    try:
        for key_id, raw in (("alice", ALICE_KEY), ("bob", BOB_KEY)):
            if db.get(APIKey, key_id) is None:
                db.add(APIKey(id=key_id, name=key_id, hashed_key=hash_api_key(raw)))
        db.commit()
        yield
    finally:
        db.close()


@pytest.fixture
def alice_cache():
    """A cache owned by alice."""
    db = SessionLocal()
    try:
        cache_id = "cachedContents/alice-secret"
        if db.get(CacheRecord, cache_id) is None:
            db.add(
                CacheRecord(
                    id=cache_id,
                    owner_id="alice",
                    model="gemini-3.5-flash",
                    display_name="alice quarterly report",
                )
            )
            db.commit()
        yield cache_id
    finally:
        db.close()


def _headers(raw_key: str) -> dict:
    return {"X-API-Key": raw_key}


def test_create_records_the_owner(client: TestClient, mock_gemini_client_global):
    """Creating a cache writes an ownership row for the calling key."""
    mock_caches = MagicMock()
    mock_gemini_client_global.caches = mock_caches
    mock_cache = MagicMock()
    mock_cache.name = "cachedContents/bob-owned"
    mock_cache.model = "gemini-3.5-flash"
    mock_cache.display_name = "bob doc"
    mock_cache.ttl = None
    mock_cache.create_time = None
    mock_cache.expire_time = None
    mock_caches.create.return_value = mock_cache

    resp = client.post(
        "/api/v1/caches/",
        json={"model": "gemini-3.5-flash", "system_instruction": "hi"},
        headers=_headers(BOB_KEY),
    )
    assert resp.status_code == 200

    db = SessionLocal()
    try:
        record = db.get(CacheRecord, "cachedContents/bob-owned")
        assert record is not None
        assert record.owner_id == "bob"
    finally:
        db.close()


def test_list_hides_another_users_cache(
    client: TestClient, mock_gemini_client_global, alice_cache
):
    """Gemini lists every cache in the project; the API must not."""
    mock_caches = MagicMock()
    mock_gemini_client_global.caches = mock_caches

    live = MagicMock()
    live.name = alice_cache
    live.model = "gemini-3.5-flash"
    live.display_name = "alice quarterly report"
    live.ttl = None
    live.create_time = None
    live.expire_time = None
    mock_caches.list.return_value = [live]

    resp = client.get("/api/v1/caches/", headers=_headers(BOB_KEY))
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_get_another_users_cache_is_forbidden(client: TestClient, alice_cache):
    resp = client.get(f"/api/v1/caches/{alice_cache}", headers=_headers(BOB_KEY))
    assert resp.status_code == 403


def test_delete_another_users_cache_is_forbidden(
    client: TestClient, mock_gemini_client_global, alice_cache
):
    mock_caches = MagicMock()
    mock_gemini_client_global.caches = mock_caches

    resp = client.delete(f"/api/v1/caches/{alice_cache}", headers=_headers(BOB_KEY))
    assert resp.status_code == 403
    mock_caches.delete.assert_not_called()

    db = SessionLocal()
    try:
        assert db.get(CacheRecord, alice_cache) is not None
    finally:
        db.close()


def test_patch_another_users_cache_is_forbidden(client: TestClient, alice_cache):
    resp = client.patch(
        f"/api/v1/caches/{alice_cache}", json={"ttl": "60s"}, headers=_headers(BOB_KEY)
    )
    assert resp.status_code == 403


def test_generate_cannot_read_another_users_cache(client: TestClient, alice_cache):
    """The leak that mattered: reusing a cache_id reads the cached documents."""
    resp = client.post(
        "/api/v1/gemini/",
        json={"model": "gemini-3.5-flash", "prompt": "summarise the document", "cache_id": alice_cache},
        headers=_headers(BOB_KEY),
    )
    assert resp.status_code == 403


def test_consensus_cannot_read_another_users_cache(client: TestClient, alice_cache):
    resp = client.post(
        "/api/v1/agents/consensus",
        json={"prompt": "what does it say?", "shared_cache_id": alice_cache},
        headers=_headers(BOB_KEY),
    )
    assert resp.status_code == 403


def test_owner_can_still_use_their_own_cache(client: TestClient, alice_cache):
    resp = client.patch(
        f"/api/v1/caches/{alice_cache}", json={"ttl": "60s"}, headers=_headers(ALICE_KEY)
    )
    assert resp.status_code == 200
