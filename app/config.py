# Do not add empty strings or values to any of the vars under Settings()

from __future__ import annotations

import os
from typing import Any

from google import genai
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.secrets import get_secret


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Platform Auth
    database_url: str | None = None
    master_api_key: str

    # Internal agent-to-agent protocol key (x-internal-key header)
    internal_api_key: str

    # Redis for rate limiting (optional)
    redis_url: str | None = None

    # Gemini API keys (optional in production — ADC via service account is used instead)
    gemini_api_key: str | None = None
    gemini_default_model: str = "gemini-2.5-flash"
    gemini_eval_model: str = "gemini-3.1-pro-preview"

    # GCP Infrastructure
    gcp_project_id: str
    gcp_region: str = "asia-southeast1"
    gcs_bucket: str | None = None
    gcp_service_account_email: str | None = None

    # Generation defaults
    default_max_output_tokens: int = 1024
    eval_max_output_tokens: int = 4096

    # Tools API keys
    alpha_vantage_api_key: str
    openweathermap_api_key: str
    news_api_key: str

    # Pinecone Configs
    pinecone_namespace: str
    pinecone_index_name: str
    pinecone_api_key: str
    gemini_embedding_model: str = "gemini-embedding-2"

    # Base URLs
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    crypto_base_url: str = "https://api.coingecko.com/api/v3"
    news_base_url: str = "https://newsapi.org/v2/everything"
    weather_base_url: str = "https://api.openweathermap.org/data/2.5/weather"
    wikipedia_base_url: str = "https://en.wikipedia.org/w/api.php"

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)

        # Env-aware embedding model: Vertex AI prod uses multimodalembedding (1408 dims);
        # Gemini API dev uses gemini-embedding-2 (3072 dims). Explicit env var overrides.
        if os.getenv("ENV") == "production" and self.gemini_embedding_model == "gemini-embedding-2":
            self.gemini_embedding_model = "multimodalembedding"

        # Attempt to fetch secrets from Secret Manager if running in production (indicated by env)
        if os.getenv("ENV") == "production":
            for field in self.model_fields:
                val = getattr(self, field)
                # If field looks like a secret, try fetching it if it's missing or empty
                if not val and field in [
                    "master_api_key",
                    "pinecone_api_key",
                    "database_url",
                    "internal_api_key",
                ]:
                    secret = get_secret(field.upper())
                    if secret:
                        setattr(self, field, secret)


settings = Settings()


def get_settings() -> Settings:
    return settings


def build_genai_client() -> genai.Client:
    if settings.gemini_api_key:
        return genai.Client(api_key=settings.gemini_api_key)
    return genai.Client(
        vertexai=True, project=settings.gcp_project_id, location=settings.gcp_region
    )

default_model = settings.gemini_default_model
eval_model = settings.gemini_eval_model
default_max_tokens = settings.default_max_output_tokens
eval_max_tokens = settings.eval_max_output_tokens
