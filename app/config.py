from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Platform Auth
    database_url: str = "sqlite:///./test.db"
    master_api_key: str = ""
    enable_execute_code: bool = False

    # Gemini API keys
    gemini_api_key: str = ""

    # GCP Infrastructure
    gcp_project_id: str = ""
    gcp_region: str = "asia-southeast1"

    # Tools API keys
    alpha_vantage_api_key: str = ""
    openweathermap_api_key: str = ""
    news_api_key: str = ""

    # Pinecone Configs
    pinecone_namespace: str = ""
    pinecone_index_name: str = ""
    pinecone_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-2"

    # Base URLs
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    crypto_base_url: str = "https://api.coingecko.com/api/v3"
    news_base_url: str = "https://newsapi.org/v2/everything"
    wandbox_base_url: str = "https://wandbox.org/api/compile.json"
    weather_base_url: str = "https://api.openweathermap.org/data/2.5/weather"
    wikipedia_base_url: str = "https://en.wikipedia.org/w/api.php"


settings = Settings()


def get_settings() -> Settings:
    return settings
