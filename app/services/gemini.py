from __future__ import annotations

import io
import json
import logging
import re
import threading
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from google.genai import types
from sqlalchemy.orm import Session

from app.config import build_genai_client, default_max_tokens, default_model
from app.database.models import UploadedFile
from app.services.llm import build_llm
from app.utils.gcs import delete_from_gcs, get_gcs_bucket_name, upload_to_gcs

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


def _log_citation_events(response: types.GenerateContentResponse, model: str) -> None:
    """Extract and log structured citation metadata from a Gemini response.

    Inspects grounding_metadata (grounding_chunks, grounding_supports,
    search_entry_point) and citation_metadata (citations with uri, title,
    start/end indices, license, publication_date) on each candidate.
    """
    for candidate in getattr(response, "candidates", []) or []:
        citation_data: dict[str, Any] = {}

        gm = getattr(candidate, "grounding_metadata", None)
        if gm:
            grounding_chunks_list = getattr(gm, "grounding_chunks", []) or []
            if grounding_chunks_list:
                chunks_data: list[dict[str, Any]] = []
                for gc in grounding_chunks_list:
                    chunk_entry: dict[str, Any] = {}
                    if getattr(gc, "web", None):
                        chunk_entry["type"] = "web"
                        chunk_entry["title"] = gc.web.title
                        chunk_entry["uri"] = gc.web.uri
                    if getattr(gc, "retrieved_context", None):
                        chunk_entry["type"] = "retrieved_context"
                        chunk_entry["title"] = gc.retrieved_context.title
                        chunk_entry["uri"] = gc.retrieved_context.uri
                    if getattr(gc, "image", None):
                        chunk_entry["type"] = "image"
                    if getattr(gc, "maps", None):
                        chunk_entry["type"] = "maps"
                    chunks_data.append(chunk_entry)
                citation_data["grounding_chunks"] = chunks_data

            grounding_supports = getattr(gm, "grounding_supports", []) or []
            if grounding_supports:
                supports_data: list[dict[str, Any]] = []
                for gs in grounding_supports:
                    segment = getattr(gs, "segment", None)
                    supports_data.append(
                        {
                            "segment_text": getattr(segment, "text", None) if segment else None,
                            "segment_start_index": (
                                getattr(segment, "start_index", None) if segment else None
                            ),
                            "segment_end_index": (
                                getattr(segment, "end_index", None) if segment else None
                            ),
                            "grounding_chunk_indices": list(
                                getattr(gs, "grounding_chunk_indices", []) or []
                            ),
                        }
                    )
                citation_data["grounding_supports"] = supports_data

            sep = getattr(gm, "search_entry_point", None)
            if sep:
                citation_data["search_entry_point"] = getattr(sep, "rendered_content", None)

        cm = getattr(candidate, "citation_metadata", None)
        if cm:
            citations_list = getattr(cm, "citations", []) or []
            if citations_list:
                cites_data: list[dict[str, Any]] = []
                for c in citations_list:
                    pub_date = getattr(c, "publication_date", None)
                    cites_data.append(
                        {
                            "uri": getattr(c, "uri", None),
                            "title": getattr(c, "title", None),
                            "start_index": getattr(c, "start_index", None),
                            "end_index": getattr(c, "end_index", None),
                            "license": getattr(c, "license", None),
                            "publication_date": str(pub_date) if pub_date else None,
                        }
                    )
                citation_data["citations"] = cites_data

        if citation_data:
            logger.info(
                json.dumps(
                    {"event": "citation", "model": model, "citation_data": citation_data},
                    default=str,
                )
            )


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

    Returns only the valid, owned attachments.  Callers should check the
    returned list length against the original request length to detect
    missing / unauthorized files.
    """
    resolved = []
    skipped = []
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
            skipped.append(att)

    if skipped:
        logger.warning(
            "%d of %d attachment(s) not found or not owned by %r: %s",
            len(skipped),
            len(attachments),
            owner_id,
            skipped,
        )
    return resolved


class _FileInfo:
    """Minimal container for uploaded file metadata. Mirrors types.File interface."""

    def __init__(self, name: str, uri: str) -> None:
        self.name = name
        self.uri = uri


def _sanitize_filename(name: str) -> str:
    """Sanitize a user-supplied filename for safe use in GCS/blob paths."""
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", name)
    sanitized = re.sub(r"\.{2,}", "_", sanitized)
    sanitized = re.sub(r"__+", "_", sanitized)
    sanitized = sanitized.strip("._- ") or "upload"
    return sanitized[:200]


def upload_file_to_gemini(file_content: bytes, display_name: str, mime_type: str) -> _FileInfo:
    """Uploads file content, routing to GCS in production or Gemini Files API in dev."""
    try:
        bucket = get_gcs_bucket_name()
        if bucket:
            logger.info(f"Uploading file '{display_name}' to GCS ({mime_type})")
            safe_name = _sanitize_filename(display_name)
            blob_name = f"{safe_name}_{uuid.uuid4().hex[:8]}"
            gcs_path = f"uploads/{blob_name}"
            upload_to_gcs(file_content, gcs_path, mime_type)
            uri = f"gs://{bucket}/{gcs_path}"
            return _FileInfo(name=gcs_path, uri=uri)

        logger.info(f"Uploading file '{display_name}' ({mime_type}) to Gemini Files API")
        file_io = io.BytesIO(file_content)
        uploaded = client.files.upload(
            file=file_io,
            config=types.UploadFileConfig(display_name=display_name, mime_type=mime_type),
        )
        name = uploaded.name or f"files/{display_name}"
        uri = uploaded.uri or ""
        return _FileInfo(name=name, uri=uri)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise


def delete_file_from_gemini(gemini_file_name: str, gemini_file_uri: str = "") -> None:
    """Deletes a file from Gemini Files API or GCS depending on URI prefix."""
    try:
        if gemini_file_name.startswith("uploads/") and gemini_file_uri.startswith("gs://"):
            logger.info(f"Deleting file '{gemini_file_name}' from GCS")
            delete_from_gcs(gemini_file_name)
            return

        logger.info(f"Deleting file '{gemini_file_name}' from Gemini Files API")
        client.files.delete(name=gemini_file_name)
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
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


def _set_request_tokens(fastapi_request: Any, usage_metadata: Any) -> None:
    """Record token counts on request.state, accumulating across multiple calls.

    Accepts either a raw genai types.GenerateContentResponseUsageMetadata
    (with prompt_token_count / candidates_token_count) or a LangChain
    UsageMetadata dict (with input_tokens / output_tokens).

    Accumulates (sums) into existing request.state values so that endpoints
    making multiple LLM calls (e.g. consensus with N workers + 1 judge)
    report the total tokens across all calls.

    Uses a threading.Lock stored on request.state to avoid races when
    concurrent run_in_threadpool() calls update the same state.
    """
    if fastapi_request is None or usage_metadata is None:
        return
    try:
        lock = getattr(fastapi_request.state, "_token_lock", None)
        if lock is None:
            lock = threading.Lock()
            fastapi_request.state._token_lock = lock

        if hasattr(usage_metadata, "prompt_token_count"):
            inp = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
            out = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        else:
            inp = int(usage_metadata.get("input_tokens", 0))
            out = int(usage_metadata.get("output_tokens", 0))

        with lock:
            prev_in = getattr(fastapi_request.state, "input_tokens", 0)
            prev_out = getattr(fastapi_request.state, "output_tokens", 0)
            fastapi_request.state.input_tokens = prev_in + inp
            fastapi_request.state.output_tokens = prev_out + out
    except Exception:
        pass  # token tracking is best-effort


def gemini_service(
    model: str,
    prompt: str,
    attachments: list[str] | None = None,
    db: Session | None = None,
    owner_id: str | None = None,
    native_tools: list[str] | None = None,
    cache_id: str | None = None,
    fastapi_request: Request | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """
    Generation service consolidated on the LangChain path.
    Reaches for raw genai.Client only when attachments or native_tools are present since LangChain's Files API integration or native tools is less direct.
    """
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError(f"max_output_tokens must be >= 1, got {max_output_tokens}")
    max_tokens = max_output_tokens if max_output_tokens is not None else default_max_tokens

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

            if cache_id and tools_config:
                logger.warning(
                    "native_tools ignored: cannot be combined with cached_content "
                    "(%s). Tool declarations must be part of the cache.",
                    cache_id,
                )
                tools_config = None

            config = types.GenerateContentConfig(
                tools=tools_config,
                safety_settings=SAFETY_SETTINGS,
                cached_content=cache_id,
                max_output_tokens=max_tokens,
            )

            # Reach for raw genai.Client for capabilities LangChain doesn't wrap
            response = client.models.generate_content(model=model, contents=contents, config=config)
            _check_safety_block(response, model)
            _log_citation_events(response, model)
            _set_request_tokens(fastapi_request, response.usage_metadata)
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
            llm = build_llm(model, max_output_tokens=max_tokens)
            llm_response = llm.invoke(prompt)
            _set_request_tokens(fastapi_request, getattr(llm_response, "usage_metadata", None))
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


def structured_service(
    model: str,
    prompt: str,
    schema: dict[str, Any],
    fastapi_request: Request | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Structured output service using raw genai.Client.
    Returns guaranteed-valid JSON matching the provided schema.
    """
    max_tokens = max_output_tokens if max_output_tokens is not None else default_max_tokens

    try:
        logger.info(f"Generating structured content with Gemini model: {model}")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                safety_settings=SAFETY_SETTINGS,
                max_output_tokens=max_tokens,
            ),
        )
        _check_safety_block(response, model)
        _set_request_tokens(fastapi_request, response.usage_metadata)

        if not response.text:
            raise ValueError("Gemini returned an empty response")

        # Parse the JSON string into a dict
        return dict(json.loads(response.text))

    except Exception as e:
        logger.error(f"Error generating structured content: {e}")
        raise


def generate_thread_title(prompt: str, model: str = default_model) -> str:
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
    max_output_tokens: int | None = None,
) -> AsyncGenerator[str, None]:
    """Streaming intentionally uses genai.Client.aio for native SSE support."""
    max_tokens = max_output_tokens if max_output_tokens is not None else default_max_tokens

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

        if cache_id and tools_config:
            logger.warning(
                "native_tools ignored: cannot be combined with cached_content "
                "(%s). Tool declarations must be part of the cache.",
                cache_id,
            )
            tools_config = None

        config = types.GenerateContentConfig(
            tools=tools_config,
            safety_settings=SAFETY_SETTINGS,
            cached_content=cache_id,
            max_output_tokens=max_tokens,
        )

        async_client = client.aio
        response = await async_client.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        sources: list[tuple[str, str]] = []
        async for chunk in response:
            if chunk.text:
                yield chunk.text

            _log_citation_events(chunk, model)

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
