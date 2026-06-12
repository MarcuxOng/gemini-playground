from __future__ import annotations

import io
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.genai import types
from sqlalchemy.orm import Session

from app.config import build_genai_client
from app.database.models import UploadedFile
from app.services.llm import build_llm

logger = logging.getLogger(__name__)
client = build_genai_client()

# Keep in sync with _SAFETY_SETTINGS dict in app/services/llm.py — update both when changing thresholds or categories so raw client and LangChain paths stay consistent.
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_LOW_AND_ABOVE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE"
    ),
]


class SafetyBlockError(Exception):
    def __init__(self, categories: list[str]) -> None:
        self.categories = categories
        super().__init__("Content blocked by safety filters")


def _check_safety_block(response: types.GenerateContentResponse, model: str) -> None:
    """Raise SafetyBlockError if the response was blocked by Gemini safety filters."""
    pf = getattr(response, "prompt_feedback", None)
    if pf and getattr(pf, "block_reason", None):
        _log_safety_block(model, ["PROMPT_BLOCKED"])
        raise SafetyBlockError(["PROMPT_BLOCKED"])

    for candidate in getattr(response, "candidates", None) or []:
        if getattr(candidate, "finish_reason", None) == types.FinishReason.SAFETY:
            ratings = getattr(candidate, "safety_ratings", None) or []
            blocked = [str(r.category) for r in ratings if getattr(r, "blocked", False)]
            _log_safety_block(model, blocked or ["UNKNOWN"])
            raise SafetyBlockError(blocked or ["UNKNOWN"])


def _log_safety_block(model: str, categories: list[str]) -> None:
    logger.warning(json.dumps({"event": "safety_block", "model": model, "categories": categories}))


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
    """Resolves DB file UUIDs to Gemini file URIs and MIME types.

    Only DB-owned UUIDs are accepted; raw URIs are rejected upstream by the
    Pydantic validators on ProviderInput and AgentRunRequest.
    """
    resolved = []
    for att in attachments:
        query = db.query(UploadedFile).filter(UploadedFile.id == att)
        if owner_id != "master":
            query = query.filter(UploadedFile.owner_id == owner_id)
        file_rec = query.first()
        if file_rec:
            resolved.append(
                {"uri": str(file_rec.gemini_file_uri), "mime_type": str(file_rec.mime_type)}
            )
        else:
            logger.warning(f"Attachment {att!r} not found or not owned by {owner_id!r}; skipping")
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


def build_native_tools(
    grounding: bool = False,
    code_exec: bool = False,
    url_context: bool = False,
    location: bool = False,
) -> list[types.Tool]:
    """Build a list of native tools for the Gemini model.

    :param grounding: Enable Google Search native tool to ground responses with web results.
    :param code_exec: Enable code_execution tool for sandboxed code evaluation.
    :param url_context: Enable url_context tool to fetch and reason over web pages.
    :param location: Enable Google Maps native tool to provide location/context via google_maps.
    """
    native_tools: list[types.Tool] = []
    if grounding:
        native_tools.append(types.Tool(google_search=types.GoogleSearch()))
    if code_exec:
        native_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
    if url_context:
        native_tools.append(types.Tool(url_context=types.UrlContext()))
    if location:
        native_tools.append(types.Tool(google_maps=types.GoogleMaps()))
    return native_tools


def gemini_service(
    model: str,
    prompt: str,
    attachments: list[str] | None = None,
    db: Session | None = None,
    owner_id: str | None = None,
    native_tools: list[str] | None = None,
    cache_id: str | None = None,
) -> str:
    """
    Generation service consolidated on the LangChain path.
    Reaches for raw genai.Client only when attachments or native_tools are present since LangChain's Files API integration or native tools is less direct.
    """
    try:
        if attachments and (not db or not owner_id):
            raise ValueError("attachments require both db and owner_id to be provided")
        if (attachments and db and owner_id) or native_tools or cache_id:
            logger.info(
                f"Generating content with attachments/native_tools using raw client: {model}"
            )
            contents: list[Any] = []
            if attachments and db and owner_id:
                resolved = resolve_attachments(attachments, db, owner_id)
                for att in resolved:
                    contents.append(
                        types.Part.from_uri(file_uri=att["uri"], mime_type=att["mime_type"])
                    )
            contents.append(prompt)

            tools_config = None
            if native_tools:
                grounding = "search" in native_tools
                code_exec = "code" in native_tools
                url_context = "url" in native_tools
                location = "location" in native_tools
                tools_config = build_native_tools(
                    grounding=grounding,
                    code_exec=code_exec,
                    url_context=url_context,
                    location=location,
                )

            config = types.GenerateContentConfig(
                tools=tools_config,
                safety_settings=SAFETY_SETTINGS,
                cached_content=cache_id,
            )

            # Reach for raw genai.Client for capabilities LangChain doesn't wrap
            response = client.models.generate_content(model=model, contents=contents, config=config)
            _check_safety_block(response, model)
            text = str(response.text or "")

            # Format and append grounding sources if search was enabled and sources found
            candidate = response.candidates[0] if response.candidates else None
            if candidate and getattr(candidate, "grounding_metadata", None):
                meta = candidate.grounding_metadata
                chunks = getattr(meta, "grounding_chunks", []) or []
                sources: list[tuple[str, str]] = []
                for chunk in chunks:
                    if chunk.web:
                        title = chunk.web.title or "Untitled"
                        uri = chunk.web.uri or ""
                        if uri and uri not in [s[1] for s in sources]:
                            sources.append((title, uri))
                if sources:
                    text += "\n\n**Sources:**\n" + "\n".join(
                        f"- [{title}]({uri})" for title, uri in sources
                    )
            return text
        else:
            logger.info(f"Generating content with Gemini model: {model}")
            llm = build_llm(model)
            llm_response = llm.invoke(prompt)
            if llm_response.response_metadata.get("finish_reason") == "SAFETY":
                blocked_categories = llm_response.response_metadata.get("safety_ratings", []) or []
                categories = [
                    str(r.get("category", "UNKNOWN"))
                    for r in blocked_categories
                    if r.get("blocked", False)
                ]
                _log_safety_block(model, categories or ["UNKNOWN"])
                raise SafetyBlockError(categories or ["UNKNOWN"])
            return str(llm_response.content)

    except Exception as e:
        logger.error(f"Error generating Gemini content: {e}")
        raise


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
                safety_settings=SAFETY_SETTINGS,
            ),
        )
        _check_safety_block(response, model)

        if not response.text:
            raise ValueError("Gemini returned an empty response")

        # Parse the JSON string into a dict
        return dict(json.loads(response.text))

    except Exception as e:
        logger.error(f"Error generating structured content: {e}")
        raise


def generate_thread_title(prompt: str, model: str = "gemini-2.5-flash") -> str:
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
    native_tools: list[str] | None = None,
    cache_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming intentionally uses genai.Client.aio for native SSE support."""
    try:
        if attachments and (not db or not owner_id):
            raise ValueError("attachments require both db and owner_id to be provided")
        logger.info(f"Starting Gemini streaming generation with model: {model}")
        contents: Any = prompt
        if (attachments and db and owner_id) or native_tools or cache_id:
            contents = []
            if attachments and db and owner_id:
                resolved = resolve_attachments(attachments, db, owner_id)
                for att in resolved:
                    contents.append(
                        types.Part.from_uri(file_uri=att["uri"], mime_type=att["mime_type"])
                    )
            contents.append(prompt)

        tools_config = None
        if native_tools:
            grounding = "search" in native_tools
            code_exec = "code" in native_tools
            url_context = "url" in native_tools
            location = "location" in native_tools
            tools_config = build_native_tools(
                grounding=grounding, code_exec=code_exec, url_context=url_context, location=location
            )

        config = types.GenerateContentConfig(
            tools=tools_config,
            safety_settings=SAFETY_SETTINGS,
            cached_content=cache_id,
        )

        async_client = client.aio
        response = await async_client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        sources: list[tuple[str, str]] = []
        async for chunk in response:
            if chunk.text:
                yield chunk.text

            # Check for grounding metadata in candidates
            for candidate in getattr(chunk, "candidates", []) or []:
                if getattr(candidate, "finish_reason", None) == types.FinishReason.SAFETY:
                    _log_safety_block(model, ["STREAM_SAFETY_BLOCK"])
                    raise SafetyBlockError(["STREAM_SAFETY_BLOCK"])
                if getattr(candidate, "grounding_metadata", None):
                    meta = candidate.grounding_metadata
                    chunks_list = getattr(meta, "grounding_chunks", []) or []
                    for c in chunks_list:
                        if c.web:
                            title = c.web.title or "Untitled"
                            uri = c.web.uri or ""
                            if uri and uri not in [s[1] for s in sources]:
                                sources.append((title, uri))

        if sources:
            source_text = "\n\n**Sources:**\n" + "\n".join(
                f"- [{title}]({uri})" for title, uri in sources
            )
            yield source_text
    except Exception as e:
        logger.error(f"Error in Gemini streaming service: {e}")
        raise
