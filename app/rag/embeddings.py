from __future__ import annotations

import logging

from google.genai import types
from langchain_core.embeddings import Embeddings

from app.config import build_genai_client, settings

logger = logging.getLogger(__name__)


class GeminiEmbeddings(Embeddings):
    """
    LangChain compatible wrapper for Google Gemini Embeddings API.
    Supports text embeddings and multimodal file embeddings.
    """

    def __init__(self, model: str | None = None):
        self.client = build_genai_client()
        self.model = model or settings.gemini_embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        try:
            response = self.client.models.embed_content(
                model=self.model, contents=texts, config={"task_type": "retrieval_document"}
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
                model=self.model, contents=text, config={"task_type": "retrieval_query"}
            )
            if response.embeddings and response.embeddings[0].values:
                return response.embeddings[0].values
            return []
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def embed_file_uri(self, file_uri: str, mime_type: str) -> list[float]:
        """Generate an embedding from a Gemini file URI using multimodal embedding.

        :param file_uri: Gemini Files API URI (e.g. 'https://...')
        :param mime_type: MIME type of the file (e.g. 'image/png', 'audio/mp3')
        :return: embedding vector as list of floats
        """
        try:
            part = types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)
            response = self.client.models.embed_content(
                model=self.model,
                contents=part,
                config={"task_type": "retrieval_document"},
            )
            if response.embeddings and response.embeddings[0].values:
                return response.embeddings[0].values
            logger.warning(f"Empty embedding returned for file URI: {file_uri}")
            return []
        except Exception as e:
            logger.error(f"Error embedding file URI {file_uri}: {e}")
            raise
