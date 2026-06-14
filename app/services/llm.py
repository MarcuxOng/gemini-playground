from __future__ import annotations

import logging
import os

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.genai.types import HarmBlockThreshold, HarmCategory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

from app.config import settings

logger = logging.getLogger(__name__)

_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

_IS_PRODUCTION = os.getenv("ENV") == "production"


def build_llm(model_name: str, temperature: float = 0.1) -> ChatVertexAI | ChatGoogleGenerativeAI:
    """Builds a Gemini LLM via Vertex AI with local fallback to Google AI Studio."""
    logger.info(f"Building Gemini LLM: {model_name}")

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
            # Proactively verify credentials to avoid lazy-init crash during invoke()
            google.auth.default()
            return ChatVertexAI(
                model=model_name,
                temperature=temperature,
                location=settings.gcp_region,
                project=project_id,
                safety_settings=_SAFETY_SETTINGS,
            )
        except DefaultCredentialsError:
            # In production, ADC must be present — no silent downgrade to AI Studio.
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

    # Fallback to Google AI Studio (local dev only — never runs in production)
    if _IS_PRODUCTION:
        raise RuntimeError(
            "GCP project ID not configured and not discoverable via ADC in production. "
            "Set GCP_PROJECT_ID env var or ensure the Cloud Run service account has ADC set up."
        )
    logger.info("Using Google AI Studio path")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        safety_settings=_SAFETY_SETTINGS,
    )
