"""Cache-backed shared context for multi-agent systems.

A SharedContext wraps a Gemini context cache and manages its lifecycle so
multiple parallel agent calls can read from a single cache_id instead of
re-sending large payloads. Solves context-bloat in concurrent scenarios.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.caches import create_context_cache, delete_cache, update_cache_ttl

logger = logging.getLogger(__name__)


@dataclass
class SharedContext:
    """A managed Gemini context cache shared across concurrent agent calls.

    Hosts a background asyncio task that refreshes the cache TTL before
    expiry so that a long-running agent swarm never hits a cold cache.

    Args:
        cache_id: The Gemini context cache resource name.
        ttl_seconds: Time-to-live in seconds (default 3600 = 1 h).
        model: The model this cache was created for.
    """

    cache_id: str
    ttl_seconds: int = 3600
    model: str = ""
    _refresh_task: asyncio.Task[None] | None = field(default=None, repr=False, init=False)

    def refresh(self) -> dict[str, Any]:
        """Synchronously extend the cache TTL by ``ttl_seconds``.

        Safe to call from any thread. Returns the updated cache metadata.
        """
        logger.info("Refreshing shared context cache %s (TTL=%ds)", self.cache_id, self.ttl_seconds)
        return update_cache_ttl(self.cache_id, f"{self.ttl_seconds}s")

    async def start_refresh_loop(self) -> None:
        """Begin a background refresh loop that extends the TTL before expiry.

        The loop fires at 80 % of *ttl_seconds*.  Call :meth:`stop_refresh`
        when the swarm session ends.
        """
        if self._refresh_task is not None and not self._refresh_task.done():
            return

        interval = min(max(int(self.ttl_seconds * 0.8), 60), self.ttl_seconds)

        async def _refresh_loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    await asyncio.to_thread(self.refresh)
                except Exception:
                    logger.exception("Failed to refresh shared context cache %s", self.cache_id)

        self._refresh_task = asyncio.create_task(_refresh_loop())
        logger.info("Started refresh loop for cache %s (interval=%ds)", self.cache_id, interval)

    def stop_refresh(self) -> None:
        """Cancel the background refresh task (no-op if not running)."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            logger.info("Stopped refresh loop for cache %s", self.cache_id)
        self._refresh_task = None

    def invalidate(self) -> None:
        """Delete the underlying Gemini context cache and stop refreshing."""
        self.stop_refresh()
        logger.info("Deleting shared context cache %s", self.cache_id)
        delete_cache(self.cache_id)

    @classmethod
    def create(
        cls,
        model: str,
        file_uris: list[str],
        mime_types: list[str],
        system_instruction: str | None = None,
        display_name: str | None = None,
        ttl_seconds: int = 3600,
    ) -> SharedContext:
        """Create a new context cache and return a managed ``SharedContext``.

        Args:
            model: Gemini model name.
            file_uris: List of Gemini file URIs (``gs://`` or Files API).
            mime_types: MIME types matching *file_uris*.
            system_instruction: Optional system-level instruction.
            display_name: Optional human label for the cache.
            ttl_seconds: Cache TTL in seconds (default 3600).
        """
        result = create_context_cache(
            model=model,
            file_uris=file_uris,
            mime_types=mime_types,
            system_instruction=system_instruction,
            display_name=display_name or "shared-context",
            ttl=f"{ttl_seconds}s",
        )
        return cls(
            cache_id=result["cache_id"],
            ttl_seconds=ttl_seconds,
            model=result.get("model", model),
        )
