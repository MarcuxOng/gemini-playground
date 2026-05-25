from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
        }
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_object)


def _gcp_credentials_available() -> bool:
    """Check for GCP credentials without making network calls.

    CloudTraceSpanExporter() will hang indefinitely when run locally without
    credentials because it falls through to the GCE metadata server, which never
    responds. This guard restricts Cloud Trace to environments where credentials
    are already resolved via a local file or a Cloud Run service account (K_SERVICE).
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    adc_file = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if adc_file.exists():
        return True
    # K_SERVICE is injected automatically by Cloud Run; metadata server is reachable there.
    if os.environ.get("K_SERVICE"):
        return True
    return False


def setup_observability(app: Any) -> None:
    if _gcp_credentials_available():
        try:
            # Configure Tracing
            tracer_provider = TracerProvider()
            cloud_trace_exporter = CloudTraceSpanExporter()  # type: ignore[no-untyped-call]
            tracer_provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))
            trace.set_tracer_provider(tracer_provider)

            # Instrument FastAPI
            FastAPIInstrumentor.instrument_app(app)
        except Exception as e:
            logger.warning(f"Observability setup skipped: {e}")
    else:
        logger.info("No GCP credentials found — Cloud Trace disabled")

    # Configure Logging
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
