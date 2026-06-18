from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


class TestSharedContext:
    def test_create_returns_shared_context(self, mock_gemini_client_global):
        from app.multi_agent.shared_context import SharedContext

        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        mock_cache = MagicMock()
        mock_cache.name = "cachedContents/shared-001"
        mock_cache.model = "gemini-2.5-flash"
        mock_cache.display_name = "shared-context"
        mock_cache.ttl = "3600s"
        mock_cache.create_time = None
        mock_cache.expire_time = None
        mock_caches.create.return_value = mock_cache

        ctx = SharedContext.create(
            model="gemini-2.5-flash",
            file_uris=[],
            mime_types=[],
            system_instruction="You are a helpful assistant.",
            ttl_seconds=3600,
        )

        assert ctx.cache_id == "cachedContents/shared-001"
        assert ctx.ttl_seconds == 3600
        assert ctx.model == "gemini-2.5-flash"

    def test_refresh_updates_ttl(self, mock_gemini_client_global):
        from app.multi_agent.shared_context import SharedContext

        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        mock_cache = MagicMock()
        mock_cache.name = "cachedContents/shared-002"
        mock_cache.model = "gemini-2.5-flash"
        mock_cache.display_name = "test"
        mock_cache.ttl = "7200s"
        mock_cache.create_time = None
        mock_cache.expire_time = None
        mock_caches.update.return_value = mock_cache

        ctx = SharedContext(cache_id="cachedContents/shared-002", ttl_seconds=7200)
        result = ctx.refresh()

        assert result["cache_id"] == "cachedContents/shared-002"
        mock_caches.update.assert_called_once()

    def test_invalidate_deletes_cache(self, mock_gemini_client_global):
        from app.multi_agent.shared_context import SharedContext

        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches

        ctx = SharedContext(cache_id="cachedContents/shared-003")
        ctx.invalidate()

        mock_caches.delete.assert_called_once_with(name="cachedContents/shared-003")

    def test_start_refresh_loop_starts_background_task(self, mock_gemini_client_global):
        import asyncio
        from app.multi_agent.shared_context import SharedContext

        mock_caches = MagicMock()
        mock_gemini_client_global.caches = mock_caches
        mock_caches.update.return_value = MagicMock(ttl="3600s")

        ctx = SharedContext(cache_id="cachedContents/shared-004", ttl_seconds=5)

        async def _run():
            await ctx.start_refresh_loop()
            assert ctx._refresh_task is not None
            ctx.stop_refresh()

        asyncio.run(_run())

    def test_stop_refresh_is_safe_when_not_running(self, mock_gemini_client_global):
        from app.multi_agent.shared_context import SharedContext

        ctx = SharedContext(cache_id="cachedContents/shared-005")
        ctx.stop_refresh()
        assert True


class TestBuildLLMWithCachedContent:
    def test_build_llm_passes_cached_content_to_chat_model(self):
        from app.services.llm import build_llm

        with patch("app.services.llm.ChatGoogleGenerativeAI") as mock_chat:
            build_llm("gemini-2.5-flash", cached_content="cachedContents/abc")

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs.get("cached_content") == "cachedContents/abc"

    def test_build_llm_without_cached_content_makes_it_none(self):
        from app.services.llm import build_llm

        with patch("app.services.llm.ChatGoogleGenerativeAI") as mock_chat:
            build_llm("gemini-2.5-flash")

        mock_chat.assert_called_once()
        assert mock_chat.call_args.kwargs.get("cached_content") is None


class TestBuildAgentWithCachedContent:
    def test_build_agent_passes_cached_content_to_build_llm(self):
        from app.agents.base import build_agent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        with patch("app.agents.base.build_llm", return_value=mock_llm) as mock_build:
            build_agent(
                tools=[],
                system_prompt="test",
                model="gemini-2.5-flash",
                cached_content="cachedContents/test-xyz",
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("cached_content") == "cachedContents/test-xyz"

    def test_build_agent_without_cached_content(self):
        from app.agents.base import build_agent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        with patch("app.agents.base.build_llm", return_value=mock_llm) as mock_build:
            build_agent(
                tools=[],
                system_prompt="test",
                model="gemini-2.5-flash",
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("cached_content") is None


class TestPresetFactoriesWithCachedContent:
    def test_coder_preset_passes_cached_content(self):
        from app.agents.presets.coder import build_coder_agent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        with patch("app.agents.base.build_llm", return_value=mock_llm) as mock_build:
            build_coder_agent(
                model="gemini-2.5-flash",
                cached_content="cachedContents/test-xyz",
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("cached_content") == "cachedContents/test-xyz"

    def test_research_preset_passes_cached_content(self):
        from app.agents.presets.research import build_research_agent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        with patch("app.agents.base.build_llm", return_value=mock_llm) as mock_build:
            build_research_agent(
                model="gemini-2.5-flash",
                cached_content="cachedContents/test-xyz",
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("cached_content") == "cachedContents/test-xyz"

    def test_analyst_preset_passes_cached_content(self):
        from app.agents.presets.analyst import build_analyst_agent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        with patch("app.agents.base.build_llm", return_value=mock_llm) as mock_build:
            build_analyst_agent(
                model="gemini-2.5-flash",
                cached_content="cachedContents/test-xyz",
            )

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("cached_content") == "cachedContents/test-xyz"

    def test_preset_without_cached_content_defaults_to_none(self):
        from app.agents.presets.coder import build_coder_agent

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")

        with patch("app.agents.base.build_llm", return_value=mock_llm) as mock_build:
            build_coder_agent(model="gemini-2.5-flash")

        mock_build.assert_called_once()
        assert mock_build.call_args.kwargs.get("cached_content") is None
