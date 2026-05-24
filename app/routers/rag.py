from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.database.models import APIKey
from app.services.rag import ingest_service, query_service
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"], dependencies=[Depends(verify_api_key)])


class IngestRequest(BaseModel):
    text: str = Field(..., max_length=100_000)


class QueryRequest(BaseModel):
    provider: str
    model: str
    query: str = Field(..., max_length=4_000)


@router.post("/ingest", response_model=APIResponse)
@limiter.limit("5/minute")
async def ingest_documents(
    request: Request, body: IngestRequest, api_key: APIKey = Depends(verify_api_key)
) -> APIResponse[Any]:
    """
    Ingest text into the Pinecone vector store.
    """
    try:
        num_chunks = await run_in_threadpool(ingest_service, body.text)
        return APIResponse(
            data={
                "message": "Successfully ingested text",
                "chunks": num_chunks,
            }
        )
    except HTTPException:
        raise
    except ValueError:
        logger.exception("Ingestion validation error")
        raise HTTPException(status_code=400, detail="Invalid ingestion request.") from None
    except Exception as e:
        logger.exception("Ingestion API error")
        raise HTTPException(status_code=500, detail="Failed to ingest text.") from e


@router.post("/query", response_model=APIResponse)
@limiter.limit("20/minute")
async def query_rag(
    request: Request, body: QueryRequest, api_key: APIKey = Depends(verify_api_key)
) -> APIResponse[Any]:
    """
    Query the RAG pipeline.
    """
    try:
        response = await run_in_threadpool(query_service, body.query, body.model, body.provider)
        return APIResponse(data={"query": body.query, "response": response})
    except HTTPException:
        raise
    except ValueError:
        logger.exception("RAG query validation error")
        raise HTTPException(status_code=400, detail="Invalid query request.") from None
    except Exception as e:
        logger.exception("RAG query API error")
        raise HTTPException(status_code=500, detail="Failed to execute RAG query.") from e
