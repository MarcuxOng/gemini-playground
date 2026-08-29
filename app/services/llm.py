from __future__ import annotations

import logging
import os
from typing import Any

import google.auth
from google.api_core.exceptions import NotFound
from google.auth.exceptions import DefaultCredentialsError
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.services.safety import LANGCHAIN_SAFETY_SETTINGS

logger = logging.getLogger(__name__)

_IS_PRODUCTION = os.getenv("ENV") == "production"

# A model absent from one Vertex endpoint surfaces as NotFound (HTTP 404) — the
# LangChain-side equivalent of the ClientError(404) the raw path retries on.
_REGION_FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (NotFound,)


def build_llm(
    model_name: str,
    temperature: float = 0.1,
    cached_content: str | None = None,
    max_output_tokens: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    thinking_budget: int | None = None,
    location: str | None = None,
) -> ChatGoogleGenerativeAI:
    """Builds a Gemini LLM via ChatGoogleGenerativeAI (Vertex AI prod, AI Studio dev).

    *location* overrides the Vertex endpoint region; it is ignored on the AI
    Studio dev path, which is not region-scoped. Pass ``"global"`` to reach
    models that only serve from the global endpoint.
    """
    logger.info(f"Building Gemini LLM: {model_name}")

    common: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "safety_settings": LANGCHAIN_SAFETY_SETTINGS,
    }
    if cached_content:
        common["cached_content"] = cached_content
    if max_output_tokens is not None:
        common["max_output_tokens"] = max_output_tokens
    if top_p is not None:
        common["top_p"] = top_p
    if top_k is not None:
        common["top_k"] = top_k
    if seed is not None:
        common["seed"] = seed
    if thinking_budget is not None:
        common["thinking_budget"] = thinking_budget

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
                project=project_id, location=location or settings.gcp_region, **common
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


def build_llm_with_region_fallback(
    model_name: str, **kwargs: Any
) -> Runnable[LanguageModelInput, BaseMessage]:
    """
    Regional-primary, global-fallback LLM for direct ``.invoke()`` call sites.

    The raw ``genai`` path already retries a 404 on the global endpoint
    (app/services/genai_calls.py); the LangChain path pinned itself to
    ``settings.gcp_region`` and had no equivalent, so a global-only model
    selected in the UI would 404 here while working on the raw path.

    Returns a plain ``ChatGoogleGenerativeAI`` on the AI Studio dev path, which
    has no regions. Agent construction deliberately keeps using ``build_llm``:
    ``create_agent`` wants a concrete chat model, not a composed Runnable.
    """
    primary = build_llm(model_name, **kwargs)

    if getattr(primary, "location", None) != settings.gcp_region:
        return primary

    fallback = build_llm(model_name, location="global", **kwargs)
    return primary.with_fallbacks([fallback], exceptions_to_handle=_REGION_FALLBACK_EXCEPTIONS)
