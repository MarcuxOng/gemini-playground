from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestCachesRouter:
    def test_create_cache_returns_200(self, client: TestClient, auth_headers, mock_gemini_client_global):
        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        mock_cache = MagicMock()
        mock_cache.name = "cachedContents/abc123"
        mock_cache.model = "gemini-2.5-flash"
        mock_cache.display_name = "test-cache"
        mock_cache.ttl = None
        mock_cache.create_time = None
        mock_cache.expire_time = None
        mock_caches.create.return_value = mock_cache

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

        mock_caches.create.assert_called_once()
        call_kwargs = mock_caches.create.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"

    def test_create_cache_requires_model_validation(self, client: TestClient, auth_headers):
        response = client.post(
            "/api/v1/caches/",
            json={"model": "gpt-4", "system_instruction": "hello"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_create_cache_requires_attachment_or_system_instruction(
        self, client: TestClient, auth_headers, mock_gemini_client_global
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

    def test_list_caches_returns_200(self, client: TestClient, auth_headers, mock_gemini_client_global):
        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches
        mock_caches.list.return_value = []

        response = client.get("/api/v1/caches/", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_get_cache_returns_200(self, client: TestClient, auth_headers, mock_gemini_client_global):
        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        mock_cache = MagicMock()
        mock_cache.name = "cachedContents/abc123"
        mock_cache.model = "gemini-2.5-flash"
        mock_cache.display_name = "test-cache"
        mock_cache.ttl = None
        mock_cache.create_time = None
        mock_cache.expire_time = None
        mock_caches.get.return_value = mock_cache

        response = client.get("/api/v1/caches/cachedContents/abc123", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["cache_id"] == "cachedContents/abc123"
        mock_caches.get.assert_called_once_with(name="cachedContents/abc123")

    def test_delete_cache_returns_200(self, client: TestClient, auth_headers, mock_gemini_client_global):
        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        response = client.delete("/api/v1/caches/cachedContents/abc123", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["data"]["deleted"] == "cachedContents/abc123"
        mock_caches.delete.assert_called_once_with(name="cachedContents/abc123")

    def test_update_cache_returns_200(self, client: TestClient, auth_headers, mock_gemini_client_global):
        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        mock_cache = MagicMock()
        mock_cache.name = "cachedContents/abc123"
        mock_cache.model = "gemini-2.5-flash"
        mock_cache.display_name = "test-cache"
        mock_cache.ttl = None
        mock_cache.create_time = None
        mock_cache.expire_time = None
        mock_caches.update.return_value = mock_cache

        response = client.patch(
            "/api/v1/caches/cachedContents/abc123",
            json={"ttl": "7200s"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["cache_id"] == "cachedContents/abc123"
        mock_caches.update.assert_called_once()
        call_kwargs = mock_caches.update.call_args.kwargs
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
