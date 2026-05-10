from __future__ import annotations

import logging

from google import genai
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddings(Embeddings):
    """
    LangChain compatible wrapper for Google Gemini Embeddings API.
    """
    def __init__(self, model: str = settings.gemini_embedding_model):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config={'task_type': 'retrieval_document'}
            )
            embeddings = []
            if response.embeddings:
                for e in response.embeddings:
                    if e.values:
                        embeddings.append(e.values)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={'task_type': 'retrieval_query'}
            )
            if response.embeddings and response.embeddings[0].values:
                return response.embeddings[0].values
            return []
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
