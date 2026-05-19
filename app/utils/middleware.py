from __future__ import annotations

import json
import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger("usage_logger")
# Ensure the logger is configured to output to stdout for Cloud Logging
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # We'll try to extract model info from the request body if it's a JSON POST
        model_name = "unknown"
        if request.method == "POST":
            try:
                # We can't easily read the body here because it might be consumed by the route handler. 
                # However, we can peek at it if needed, or rely on the route handler to attach info to the request state.
                pass
            except Exception:
                pass

        response = await call_next(request)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Get info from request state (to be populated by route handlers or auth dependencies)
        api_key_id = getattr(request.state, "api_key_id", "anonymous")
        model = getattr(request.state, "model", model_name)
        input_tokens = getattr(request.state, "input_tokens", 0)
        output_tokens = getattr(request.state, "output_tokens", 0)

        log_data = {
            "event": "api_request",
            "api_key_id": api_key_id,
            "route": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        logger.info(json.dumps(log_data))

        return response
