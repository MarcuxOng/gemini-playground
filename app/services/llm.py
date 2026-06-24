from __future__ import annotations

import logging
import os
from typing import Any

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.genai.types import HarmBlockThreshold, HarmCategory
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

logger = logging.getLogger(__name__)

# Dict format used by ChatGoogleGenerativeAI (both Vertex AI and AI Studio modes).
# Keep in sync with SAFETY_SETTINGS list in app/services/gemini.py.
_SAFETY_SETTINGS: dict[HarmCategory, HarmBlockThreshold] = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

_IS_PRODUCTION = os.getenv("ENV") == "production"


def build_llm(
    model_name: str,
    temperature: float = 0.1,
    cached_content: str | None = None,
    max_output_tokens: int | None = None,
) -> ChatGoogleGenerativeAI:
    """Builds a Gemini LLM via ChatGoogleGenerativeAI (Vertex AI prod, AI Studio dev)."""
    logger.info(f"Building Gemini LLM: {model_name}")

    common: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "safety_settings": _SAFETY_SETTINGS,
    }
    if cached_content:
        common["cached_content"] = cached_content
    if max_output_tokens is not None:
        common["max_output_tokens"] = max_output_tokens

    project_id: str | None = None
    if settings.gcp_project_id:
        project_id = settings.gcp_project_id

    if not project_id and _IS_PRODUCTION:
        try:
            _, adc_project_id = google.auth.default()
            project_id = adc_project_id
            if project_id:
                logger.info(f"Using ADC-discovered project ID: {project_id}")
        except Exception:
            pass

    if project_id:
        try:
            google.auth.default()
            return ChatGoogleGenerativeAI(
                project=project_id, location=settings.gcp_region, **common
            )
        except DefaultCredentialsError:
            if _IS_PRODUCTION:
                raise RuntimeError(
                    "ADC credentials not found in production. "
                    "Ensure the Cloud Run service account has roles/aiplatform.user."
                ) from None
            logger.warning("Application Default Credentials (ADC) not found. Falling back.")
        except Exception as e:
            if _IS_PRODUCTION:
                raise
            logger.warning(f"Vertex AI initialization check failed: {e}")

    if _IS_PRODUCTION:
        raise RuntimeError(
            "GCP project ID not configured and not discoverable via ADC in production. "
            "Set GCP_PROJECT_ID env var or ensure the Cloud Run service account has ADC set up."
        )
    logger.info("Using Google AI Studio path")
    return ChatGoogleGenerativeAI(google_api_key=settings.gemini_api_key, **common)
