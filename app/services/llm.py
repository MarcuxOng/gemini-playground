from __future__ import annotations

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

from app.config import settings

logger = logging.getLogger(__name__)


def build_llm(model_name: str, temperature: float = 0.1) -> ChatVertexAI | ChatGoogleGenerativeAI:
    """Builds a Gemini LLM via Vertex AI with local fallback to Google AI Studio."""
    logger.info(f"Building Gemini LLM: {model_name}")
    try:
        return ChatVertexAI(
            model=model_name,
            temperature=temperature,
            location=settings.gcp_region,
            project=settings.gcp_project_id,
        )
    except Exception as e:
        logger.warning(f"Vertex AI initialization failed, falling back to Google AI Studio: {e}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
        )