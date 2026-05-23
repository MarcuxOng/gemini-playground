from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.database.db import get_db
from app.database.models import APIKey, UploadedFile
from app.services.gemini import delete_file_from_gemini, upload_file_to_gemini
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/files", tags=["Files"], dependencies=[Depends(verify_api_key)])


class FileResponse(BaseModel):
    id: str
    gemini_file_name: str
    gemini_file_uri: str
    mime_type: str
    size_bytes: int
    display_name: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.post("/upload", response_model=APIResponse[FileResponse])
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[FileResponse]:
    """Upload a file to the Gemini Files API and track it locally."""
    try:
        content = await file.read()
        size_bytes = len(content)
        display_name = file.filename or "unknown"
        mime_type = file.content_type or "application/octet-stream"

        # Upload to Gemini Files API
        uploaded = await run_in_threadpool(
            upload_file_to_gemini,
            file_content=content,
            display_name=display_name,
            mime_type=mime_type,
        )

        # Create local DB record
        new_file = UploadedFile(
            gemini_file_name=uploaded.name,
            gemini_file_uri=uploaded.uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            display_name=display_name,
            owner_id=api_key.id,
        )
        db.add(new_file)
        db.commit()
        db.refresh(new_file)

        logger.info(
            f"File uploaded and tracked. DB ID: {new_file.id}, Gemini Name: {new_file.gemini_file_name}"
        )

        response_data = FileResponse.model_validate(new_file)
        return APIResponse(data=response_data)

    except Exception as e:
        logger.exception("Failed to upload file")
        raise HTTPException(status_code=500, detail="Upload failed") from e


@router.get("/", response_model=APIResponse[list[FileResponse]])
@limiter.limit("30/minute")
async def list_files(
    request: Request,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[list[FileResponse]]:
    """List all files uploaded by the authenticated user."""
    try:
        query = db.query(UploadedFile)
        if api_key.id != "master":
            query = query.filter(UploadedFile.owner_id == api_key.id)
        files = query.order_by(UploadedFile.created_at.desc()).all()
        response_data = [FileResponse.model_validate(f) for f in files]
        return APIResponse(data=response_data)
    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail="Failed to list files") from e


@router.delete("/{file_id}", response_model=APIResponse[dict[str, str]])
@limiter.limit("10/minute")
async def delete_file(
    request: Request,
    file_id: str,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse[dict[str, str]]:
    """Delete a file from the database and Gemini Files API."""
    try:
        query = db.query(UploadedFile).filter(UploadedFile.id == file_id)
        if api_key.id != "master":
            query = query.filter(UploadedFile.owner_id == api_key.id)
        file_rec = query.first()

        if not file_rec:
            raise HTTPException(status_code=404, detail="File not found or access denied.")

        # Delete from DB first to avoid orphaned rows if commit fails
        gemini_file_name = str(file_rec.gemini_file_name)
        db.delete(file_rec)
        db.commit()

        # Delete from Gemini Files API (best-effort after DB commit)
        try:
            await run_in_threadpool(delete_file_from_gemini, gemini_file_name=gemini_file_name)
        except Exception:
            logger.exception(
                f"Failed to delete Gemini file {gemini_file_name}; DB record already removed"
            )

        logger.info(f"Deleted file record {file_id} (Gemini: {gemini_file_name})")
        return APIResponse(data={"message": f"Successfully deleted file {file_id}"})

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete file {file_id}")
        raise HTTPException(status_code=500, detail="Failed to delete file") from e
