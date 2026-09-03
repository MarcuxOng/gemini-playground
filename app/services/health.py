"""Dependency probes behind ``/api/v1/health`` (T13).

The endpoint is public and polled — by uptime monitors, and by Cloud Run itself — so
every probe here is a metadata call that spends no tokens, they run concurrently
rather than in series, each is capped by its own timeout, and the assembled report is
cached for a few seconds so a hot loop cannot fan out to three backends per request.

Failure detail is the exception *type* only. Connection errors routinely carry the DSN
or host in their message, and this response is unauthenticated.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Literal

from pinecone import Pinecone
from pydantic import BaseModel
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from app.config import client, settings
from app.database.db import SessionLocal
from app.utils.limiter import limiter_storage_status

logger = logging.getLogger(__name__)

Status = Literal["ok", "degraded", "down"]

# A probe that has not answered in this long is reported as down rather than held open.
# `run_in_threadpool` cannot actually be cancelled, so the worker thread runs to
# completion in the background — the timeout bounds the *response*, not the probe.
_PROBE_TIMEOUT_SECONDS = 3.0

# Serve repeated polls from one set of probes. Uptime monitors, load balancers and
# Cloud Run all hit this path on their own schedules, and it is the one route with no
# API key in front of it.
_CACHE_TTL_SECONDS = 5.0

_cached: tuple[float, HealthReport] | None = None


class DependencyHealth(BaseModel):
    """One dependency's verdict.

    ``critical`` is why the overall verdict is not simply "everything must be ok".
    A downed Pinecone breaks RAG while the MCP tool set — the product under Option B —
    keeps answering, and Decision 1 already ruled that an unreachable rate-limit store
    degrades rather than taking the surface down. Reporting either as a hard failure
    would make this endpoint contradict a decision the project has already made.
    """

    status: Status
    critical: bool
    detail: str | None = None


class HealthReport(BaseModel):
    status: Status
    dependencies: dict[str, DependencyHealth]

    @property
    def http_status_code(self) -> int:
        return 503 if self.status == "down" else 200


def _probe_database() -> DependencyHealth:
    """Postgres holds the API keys, so losing it closes every authenticated route."""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
    finally:
        db.close()
    return DependencyHealth(status="ok", critical=True)


def _probe_gemini() -> DependencyHealth:
    """Fetch the default model's metadata — no generation, no tokens.

    Targeting the configured default rather than listing models means this also
    catches the region split (T4): a model that serves from `global` but 404s on
    `us-central1` fails here instead of at the first real call.
    """
    client.models.get(model=settings.gemini_default_model)
    return DependencyHealth(status="ok", critical=True)


def _probe_pinecone() -> DependencyHealth:
    """Index stats, not a query — no embedding call, so nothing is spent.

    Not critical: RAG is one router, and none of the §7 tool set goes through it.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    pc.Index(settings.pinecone_index_name).describe_index_stats()
    return DependencyHealth(status="ok", critical=False)


def _probe_rate_limit_store() -> DependencyHealth:
    """Never reports ``down`` — Decision 1 made this dependency non-fatal by design.

    `reachable` goes false the moment the store stops answering; `degraded` only
    flips once traffic has actually failed over to in-memory counters. Both are worth
    reporting, and neither is an outage: the API is up and still limiting, just
    per-instance.
    """
    status = limiter_storage_status()
    if status["reachable"] and not status["degraded"]:
        return DependencyHealth(status="ok", critical=False, detail=str(status["backend"]))

    detail = (
        "unreachable; limits are per-instance until it recovers"
        if not status["reachable"]
        else "failed over to in-memory counters; probing for recovery"
    )
    return DependencyHealth(status="degraded", critical=False, detail=detail)


async def _run_probe(
    name: str, probe: Callable[[], DependencyHealth], critical: bool
) -> tuple[str, DependencyHealth]:
    try:
        result = await asyncio.wait_for(run_in_threadpool(probe), _PROBE_TIMEOUT_SECONDS)
        return name, result
    except TimeoutError:
        logger.warning("Health probe '%s' timed out after %ss.", name, _PROBE_TIMEOUT_SECONDS)
        return name, DependencyHealth(
            status="down", critical=critical, detail=f"timed out after {_PROBE_TIMEOUT_SECONDS}s"
        )
    except Exception as exc:
        # Type only. The message frequently carries the host or DSN, and this
        # response goes out without an API key in front of it.
        logger.warning("Health probe '%s' failed: %s", name, type(exc).__name__, exc_info=True)
        return name, DependencyHealth(status="down", critical=critical, detail=type(exc).__name__)


async def check_health(use_cache: bool = True) -> HealthReport:
    """Probe every dependency concurrently and assemble the overall verdict."""
    global _cached

    if use_cache and _cached is not None:
        cached_at, report = _cached
        if (time.monotonic() - cached_at) < _CACHE_TTL_SECONDS:
            return report

    probes: list[tuple[str, Callable[[], DependencyHealth], bool]] = [
        ("database", _probe_database, True),
        ("gemini", _probe_gemini, True),
        ("pinecone", _probe_pinecone, False),
        ("rate_limit_store", _probe_rate_limit_store, False),
    ]

    results = await asyncio.gather(*(_run_probe(*probe) for probe in probes))
    dependencies = dict(results)

    if any(dep.status == "down" and dep.critical for dep in dependencies.values()):
        overall: Status = "down"
    elif any(dep.status != "ok" for dep in dependencies.values()):
        overall = "degraded"
    else:
        overall = "ok"

    report = HealthReport(status=overall, dependencies=dependencies)
    # Two concurrent first-hits can both probe. Harmless, and cheaper than a lock.
    _cached = (time.monotonic(), report)
    return report
