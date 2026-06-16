from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.database.db import get_db
from app.database.models import APIKey
from app.services.rag import ingest_file_service, ingest_service, query_service
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse
from app.utils.sanitizer import sanitize_prompt
from app.utils.validators import ModelName

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"], dependencies=[Depends(verify_api_key)])


class IngestRequest(BaseModel):
    text: str = Field(None, max_length=100_000)  # type: ignore[assignment]
    file_ids: list[str] | None = None


class QueryRequest(BaseModel):
    model: ModelName = "gemini-2.5-flash"
    query: str = Field(..., max_length=4_000)


@router.post("/ingest", response_model=APIResponse)
@limiter.limit("5/minute")
async def ingest_documents(
    request: Request,
    body: IngestRequest,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[Any]:
    """
    Ingest text and/or multimodal files into the Pinecone vector store.

    Provide `text` for plain-text chunks, `file_ids` for previously uploaded
    Gemini files (image, audio, video, PDF) that should be embedded via
    multimodal embedding and stored for later retrieval.
    """
    try:
        text_chunks = 0
        file_count = 0
        owner_key = str(api_key.id)

        if body.text:
            text = sanitize_prompt(body.text)
            text_chunks = await run_in_threadpool(ingest_service, text, owner_id=owner_key)

        if body.file_ids:
            file_count = await run_in_threadpool(ingest_file_service, body.file_ids, db, owner_key)

        if not text_chunks and not file_count:
            raise ValueError("Either 'text' or 'file_ids' must be provided")

        return APIResponse(
            data={
                "message": "Successfully ingested content",
                "text_chunks": text_chunks,
                "files": file_count,
            }
        )
    except HTTPException:
        raise
    except ValueError:
        logger.exception("Ingestion validation error")
        raise HTTPException(status_code=400, detail="Invalid ingestion request.") from None
    except Exception as e:
        logger.exception("Ingestion API error")
        raise HTTPException(status_code=500, detail="Failed to ingest content.") from e


@router.post("/query", response_model=APIResponse)
@limiter.limit("20/minute")
async def query_rag(
    request: Request, body: QueryRequest, api_key: APIKey = Depends(verify_api_key)
) -> APIResponse[Any]:
    """
    Query the RAG pipeline.
    """
    try:
        query = sanitize_prompt(body.query)
        owner_key = str(api_key.id)
        response = await run_in_threadpool(
            query_service, query, body.model, owner_id=owner_key
        )
        return APIResponse(data={"query": body.query, "response": response})
    except HTTPException:
        raise
    except ValueError:
        logger.exception("RAG query validation error")
        raise HTTPException(status_code=400, detail="Invalid query request.") from None
    except Exception as e:
        logger.exception("RAG query API error")
        raise HTTPException(status_code=500, detail="Failed to execute RAG query.") from e
