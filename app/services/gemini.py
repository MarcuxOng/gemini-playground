from __future__ import annotations

import io
import json
import logging
import mimetypes
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import UploadedFile
from app.services.llm import build_llm

logger = logging.getLogger(__name__)
client = genai.Client(api_key=settings.gemini_api_key)


def list_gemini_models() -> list[str]:
    try:
        logger.info("Fetching Gemini models...")
        models = sorted(client.models.list(), key=lambda m: m.name or "")
        model_list: list[str] = []
        for m in models:
            if m.name:
                response = m.name.replace("models/", "")
                model_list.append(response)

        return model_list

    except Exception as e:
        logger.error(f"Error fetching Gemini models: {e}")
        raise


def resolve_attachments(attachments: list[str], db: Session, owner_id: str) -> list[dict[str, str]]:
    """Resolves attachment IDs or URIs to file URIs and MIME types."""
    resolved = []
    for att in attachments:
        is_uuid = False
        try:
            uuid.UUID(att)
            is_uuid = True
        except ValueError:
            pass

        if is_uuid:
            query = db.query(UploadedFile).filter(UploadedFile.id == att)
            if owner_id != "master":
                query = query.filter(UploadedFile.owner_id == owner_id)
            file_rec = query.first()
            if file_rec:
                resolved.append({"uri": str(file_rec.gemini_file_uri), "mime_type": str(file_rec.mime_type)})
        elif att.startswith("https://") or att.startswith("gs://"):
            mime_type, _ = mimetypes.guess_type(att)
            resolved.append({"uri": att, "mime_type": mime_type or "application/octet-stream"})
    return resolved


def upload_file_to_gemini(file_content: bytes, display_name: str, mime_type: str) -> types.File:
    """Uploads file content directly to Gemini Files API."""
    try:
        logger.info(f"Uploading file '{display_name}' ({mime_type}) to Gemini Files API")
        file_io = io.BytesIO(file_content)
        uploaded = client.files.upload(
            file=file_io,
            config=types.UploadFileConfig(display_name=display_name, mime_type=mime_type),
        )
        return uploaded
    except Exception as e:
        logger.error(f"Error uploading file to Gemini Files API: {e}")
        raise


def delete_file_from_gemini(gemini_file_name: str) -> None:
    """Deletes file from Gemini Files API."""
    try:
        logger.info(f"Deleting file '{gemini_file_name}' from Gemini Files API")
        client.files.delete(name=gemini_file_name)
    except Exception as e:
        logger.error(f"Error deleting file from Gemini Files API: {e}")
        raise



def gemini_service(
    model: str,
    prompt: str,
    attachments: list[str] | None = None,
    db: Session | None = None,
    owner_id: str | None = None,
) -> str:
    """
    Generation service consolidated on the LangChain path.
    Reaches for raw genai.Client only when attachments are present since LangChain's Files API integration is less direct.
    """
    try:
        if attachments and db and owner_id:
            logger.info(
                f"Generating content with attachments using raw client: {model}"
            )
            contents: list[Any] = []
            resolved = resolve_attachments(attachments, db, owner_id)
            for att in resolved:
                contents.append(
                    types.Part.from_uri(file_uri=att["uri"], mime_type=att["mime_type"])
                )
            contents.append(prompt)

            # Reach for raw genai.Client for capabilities LangChain doesn't wrap
            response = client.models.generate_content(model=model, contents=contents)
            return str(response.text or "")
        else:
            logger.info(f"Generating content with Gemini model: {model}")
            llm = build_llm(model)
            llm_response = llm.invoke(prompt)
            return str(llm_response.content)

    except Exception as e:
        logger.error(f"Error generating Gemini content: {e}")
        raise (e)


def structured_service(model: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    """
    Structured output service using raw genai.Client.
    Returns guaranteed-valid JSON matching the provided schema.
    """
    try:
        logger.info(f"Generating structured content with Gemini model: {model}")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response")

        # Parse the JSON string into a dict
        return dict(json.loads(response.text))

    except Exception as e:
        logger.error(f"Error generating structured content: {e}")
        raise (e)


def generate_thread_title(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """
    Generates a short (3-5 words) descriptive title for a thread based on the initial prompt.
    """
    try:
        logger.info("Generating thread title...")
        # Use a concise internal prompt for title generation
        title_prompt = (
            f"Generate a concise, 3-5 word title for a conversation that starts with: '{prompt}'. "
            "Respond ONLY with the title text, no quotes or punctuation."
        )
        llm = build_llm(model)
        response = llm.invoke(title_prompt)
        title = str(response.content).strip()
        # Clean up any quotes if the model ignored instructions
        return title.replace('"', "").replace("'", "")
    except Exception as e:
        logger.error(f"Error generating thread title: {e}")
        # Fallback to a truncated version of the prompt if LLM fails
        return prompt[:30] + "..." if len(prompt) > 30 else prompt


async def gemini_stream_service(
    model: str,
    prompt: str,
    attachments: list[str] | None = None,
    db: Session | None = None,
    owner_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming intentionally uses genai.Client.aio for native SSE support."""
    try:
        logger.info(f"Starting Gemini streaming generation with model: {model}")
        contents: Any = prompt
        if attachments and db and owner_id:
            contents = []
            resolved = resolve_attachments(attachments, db, owner_id)
            for att in resolved:
                contents.append(
                    types.Part.from_uri(file_uri=att["uri"], mime_type=att["mime_type"])
                )
            contents.append(prompt)

        async with client.aio as async_client:
            response = await async_client.models.generate_content_stream(
                model=model, contents=contents
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
    except Exception as e:
        logger.error(f"Error in Gemini streaming service: {e}")
        raise (e)
