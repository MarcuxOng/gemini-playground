from __future__ import annotations

from fastapi import APIRouter, Response

from app.services.health import check_health
from app.utils.response import APIResponse

# Deliberately no `verify_api_key` dependency and no rate limit — health and root are
# the only public routes. What keeps that safe is the probe cache in
# app/services/health.py: an unauthenticated caller cannot turn a request loop into a
# fan-out against Postgres, Vertex and Pinecone.
router = APIRouter(prefix="/api/v1/health", tags=["Health"])


@router.get("", response_model=APIResponse)
async def health(response: Response) -> APIResponse:  # type: ignore[type-arg]
    """Report each dependency's status and an overall verdict.

    Returns 503 naming the dependency when a *critical* one is down. A degraded
    rate-limit store or an unreachable Pinecone is reported in the body but still
    answers 200 — see `DependencyHealth.critical` for why.
    """
    report = await check_health()
    response.status_code = report.http_status_code

    error = None
    if report.status == "down":
        # `success=False` and this message are reserved for the 503. A non-critical
        # dependency that is down still shows up under `data.dependencies`, but it is
        # not an outage and must not be reported as one.
        down = [name for name, dep in report.dependencies.items() if dep.status == "down"]
        error = f"Dependencies unavailable: {', '.join(down)}"

    return APIResponse(
        success=report.status != "down",
        data=report.model_dump(),
        error=error,
    )
