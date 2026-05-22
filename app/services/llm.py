from __future__ import annotations

import logging

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

from app.config import settings

logger = logging.getLogger(__name__)


def build_llm(model_name: str, temperature: float = 0.1) -> ChatVertexAI | ChatGoogleGenerativeAI:
    """Builds a Gemini LLM via Vertex AI with local fallback to Google AI Studio."""
    logger.info(f"Building Gemini LLM: {model_name}")

    # If we have a project ID, try Vertex AI
    if settings.gcp_project_id:
        try:
            # Proactively verify credentials to avoid lazy-init crash during invoke()
            google.auth.default()
            return ChatVertexAI(
                model=model_name,
                temperature=temperature,
                location=settings.gcp_region,
                project=settings.gcp_project_id,
            )
        except DefaultCredentialsError:
            logger.warning("Application Default Credentials (ADC) not found. Falling back.")
        except Exception as e:
            logger.warning(f"Vertex AI initialization check failed: {e}")

    # Fallback to Google AI Studio (requires GEMINI_API_KEY)
    logger.info("Using Google AI Studio path")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )
