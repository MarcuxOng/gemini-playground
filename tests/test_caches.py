from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.database.db import SessionLocal
from app.database.models import APIKey, CacheRecord
from app.utils.auth import hash_api_key

ALICE_KEY = "alice-plaintext-key"
BOB_KEY = "bob-plaintext-key"


class TestCachesRouter:
    def test_create_cache_returns_200(self, client: TestClient, auth_headers, gemini_caches, make_gemini_cache):
        gemini_caches.create.return_value = make_gemini_cache(name="cachedContents/abc123")

        response = client.post(
            "/api/v1/caches/",
            json={
                "model": "gemini-2.5-flash",
                "system_instruction": "You are a helpful assistant.",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["cache_id"] == "cachedContents/abc123"
        assert data["model"] == "gemini-2.5-flash"

        gemini_caches.create.assert_called_once()
        call_kwargs = gemini_caches.create.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"

    def test_create_cache_requires_model_validation(self, client: TestClient, auth_headers):
        response = client.post(
            "/api/v1/caches/",
            json={"model": "gpt-4", "system_instruction": "hello"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_create_cache_requires_attachment_or_system_instruction(
        self, client: TestClient, auth_headers
    ):
        response = client.post(
            "/api/v1/caches/",
            json={"model": "gemini-2.5-flash"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_create_cache_validates_attachment_ids(self, client: TestClient, auth_headers):
        response = client.post(
            "/api/v1/caches/",
            json={
                "model": "gemini-2.5-flash",
                "attachments": ["not-a-uuid"],
                "system_instruction": "test",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_create_cache_validates_ttl_pattern(self, client: TestClient, auth_headers):
        response = client.post(
            "/api/v1/caches/",
            json={
                "model": "gemini-2.5-flash",
                "system_instruction": "test",
                "ttl": "invalid",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_list_caches_returns_200(self, client: TestClient, auth_headers, gemini_caches):
        gemini_caches.list.return_value = []

        response = client.get("/api/v1/caches/", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_get_cache_returns_200(self, client: TestClient, auth_headers, gemini_caches, make_gemini_cache):
        gemini_caches.get.return_value = make_gemini_cache(name="cachedContents/abc123")

        response = client.get("/api/v1/caches/cachedContents/abc123", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["cache_id"] == "cachedContents/abc123"
        gemini_caches.get.assert_called_once_with(name="cachedContents/abc123")

    def test_delete_cache_returns_200(self, client: TestClient, auth_headers, gemini_caches):
        response = client.delete("/api/v1/caches/cachedContents/abc123", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["data"]["deleted"] == "cachedContents/abc123"
        gemini_caches.delete.assert_called_once_with(name="cachedContents/abc123")

    def test_update_cache_returns_200(self, client: TestClient, auth_headers, gemini_caches, make_gemini_cache):
        gemini_caches.update.return_value = make_gemini_cache(name="cachedContents/abc123")

        response = client.patch(
            "/api/v1/caches/cachedContents/abc123",
            json={"ttl": "7200s"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["cache_id"] == "cachedContents/abc123"
        gemini_caches.update.assert_called_once()
        call_kwargs = gemini_caches.update.call_args.kwargs
        assert call_kwargs["name"] == "cachedContents/abc123"

    def test_update_cache_validates_ttl(self, client: TestClient, auth_headers):
        response = client.patch(
            "/api/v1/caches/cachedContents/abc123",
            json={"ttl": "not-a-ttl"},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestGeminiWithCacheId:
    def test_gemini_with_cache_id_passes_to_service(
        self, client: TestClient, auth_headers, mock_gemini_client_global
    ):
        mock_response = MagicMock()
        mock_response.text = "cached response"
        mock_response.candidates = []
        mock_response.prompt_feedback = None

        with patch.object(
            mock_gemini_client_global.models,
            "generate_content",
            return_value=mock_response,
        ) as mock_gen:
            response = client.post(
                "/api/v1/gemini/",
                json={
                    "model": "gemini-2.5-flash",
                    "prompt": "What does the document say?",
                    "cache_id": "cachedContents/abc123",
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data == "cached response"

        call_kwargs = mock_gen.call_args.kwargs
        config = call_kwargs["config"]
        assert config.cached_content == "cachedContents/abc123"

    def test_gemini_without_cache_id_omits_cached_content(
        self, client: TestClient, auth_headers, mock_gemini_client_global
    ):
        mock_response = MagicMock()
        mock_response.text = '{"result": "ok"}'
        mock_response.candidates = []
        mock_response.prompt_feedback = None

        with patch.object(
            mock_gemini_client_global.models, "generate_content", return_value=mock_response
        ) as mock_gen:
            response = client.post(
                "/api/v1/gemini/",
                json={
                    "model": "gemini-2.5-flash",
                    "prompt": "hello",
                    "attachments": ["00000000-0000-0000-0000-000000000001"],
                },
                headers=auth_headers,
            )

        assert response.status_code == 200
        body = response.json()
        assert body["data"] == '{"result": "ok"}'

        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs["config"].cached_content is None

    def test_stream_with_cache_id_passes_to_service(
        self, client: TestClient, auth_headers, mock_gemini_client_global
    ):
        from unittest.mock import AsyncMock

        mock_aio = MagicMock()
        mock_gemini_client_global.aio = mock_aio

        mock_generate = AsyncMock()
        mock_aio.models.generate_content_stream = mock_generate

        async def mock_stream_gen():
            mock_chunk = MagicMock()
            mock_chunk.text = "cached stream response"
            mock_chunk.candidates = []
            yield mock_chunk

        mock_generate.return_value = mock_stream_gen()

        response = client.post(
            "/api/v1/gemini/stream",
            json={
                "model": "gemini-2.5-flash",
                "prompt": "What does the document say?",
                "cache_id": "cachedContents/abc123",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        content_str = response.content.decode("utf-8")
        assert "cached stream response" in content_str

        call_kwargs = mock_generate.call_args.kwargs
        config = call_kwargs["config"]
        assert config.cached_content == "cachedContents/abc123"

    def test_caches_endpoint_requires_auth(self, client: TestClient):
        response = client.post(
            "/api/v1/caches/",
            json={"model": "gemini-2.5-flash", "system_instruction": "test"},
        )
        assert response.status_code in [401, 422]


class TestCacheOwnership:
    """Context caches are project-scoped Gemini resources with no owner of their own.

    Before ``playground_v1_caches`` existed, every authenticated caller could list,
    read, re-use and delete every other caller's cache — and a cache holds the
    *contents* of the documents it was built from. These tests pin that shut.
    """

    @pytest.fixture(scope="class", autouse=True)
    def two_users(self):
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

    @pytest.fixture()
    def alice_cache(self):
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

    def test_create_records_the_owner(self, client: TestClient, gemini_caches, make_gemini_cache):
        """Creating a cache writes an ownership row for the calling key."""
        gemini_caches.create.return_value = make_gemini_cache(
            name="cachedContents/bob-owned", display_name="bob doc"
        )

        resp = client.post(
            "/api/v1/caches/",
            json={"model": "gemini-3.5-flash", "system_instruction": "hi"},
            headers={"X-API-Key": BOB_KEY},
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
        self, client: TestClient, gemini_caches, make_gemini_cache, alice_cache
    ):
        """Gemini lists every cache in the project; the API must not."""
        gemini_caches.list.return_value = [
            make_gemini_cache(name=alice_cache, display_name="alice quarterly report")
        ]

        resp = client.get("/api/v1/caches/", headers={"X-API-Key": BOB_KEY})
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_get_another_users_cache_is_forbidden(self, client: TestClient, alice_cache):
        resp = client.get(f"/api/v1/caches/{alice_cache}", headers={"X-API-Key": BOB_KEY})
        assert resp.status_code == 403

    def test_delete_another_users_cache_is_forbidden(
        self, client: TestClient, gemini_caches, alice_cache
    ):
        resp = client.delete(f"/api/v1/caches/{alice_cache}", headers={"X-API-Key": BOB_KEY})
        assert resp.status_code == 403
        gemini_caches.delete.assert_not_called()

        db = SessionLocal()
        try:
            assert db.get(CacheRecord, alice_cache) is not None
        finally:
            db.close()

    def test_patch_another_users_cache_is_forbidden(self, client: TestClient, alice_cache):
        resp = client.patch(
            f"/api/v1/caches/{alice_cache}", json={"ttl": "60s"}, headers={"X-API-Key": BOB_KEY}
        )
        assert resp.status_code == 403

    def test_generate_cannot_read_another_users_cache(self, client: TestClient, alice_cache):
        """The leak that mattered: reusing a cache_id reads the cached documents."""
        resp = client.post(
            "/api/v1/gemini/",
            json={"model": "gemini-3.5-flash", "prompt": "summarise the document", "cache_id": alice_cache},
            headers={"X-API-Key": BOB_KEY},
        )
        assert resp.status_code == 403

    def test_consensus_cannot_read_another_users_cache(self, client: TestClient, alice_cache):
        resp = client.post(
            "/api/v1/agents/consensus",
            json={"prompt": "what does it say?", "shared_cache_id": alice_cache},
            headers={"X-API-Key": BOB_KEY},
        )
        assert resp.status_code == 403

    def test_owner_can_still_use_their_own_cache(self, client: TestClient, alice_cache):
        resp = client.patch(
            f"/api/v1/caches/{alice_cache}", json={"ttl": "60s"}, headers={"X-API-Key": ALICE_KEY}
        )
        assert resp.status_code == 200
