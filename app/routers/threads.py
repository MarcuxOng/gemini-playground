from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Thread
from app.utils.auth import verify_master_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

router = APIRouter(
    prefix="/api/v1/threads", tags=["Threads"], dependencies=[Depends(verify_master_key)]
)


@router.get("/", response_model=APIResponse)
@limiter.limit("30/minute")
async def list_threads(
    request: Request,
    db: Session = Depends(get_db),
) -> APIResponse:  # type: ignore[type-arg]
    threads = db.query(Thread).order_by(Thread.updated_at.desc()).all()
    return APIResponse(
        data=[
            {
                "id": t.id,
                "title": t.title,
                "preset": t.preset,
                "model": t.model,
                "created_at": t.created_at,
            }
            for t in threads
        ]
    )


@router.get("/{thread_id}/messages", response_model=APIResponse)
@limiter.limit("30/minute")
async def get_thread_messages(
    request: Request,
    thread_id: str,
    db: Session = Depends(get_db),
) -> APIResponse:  # type: ignore[type-arg]
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(404, "Thread not found.")
    return APIResponse(
        data=[
            {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}
            for m in thread.messages
        ]
    )


@router.delete("/{thread_id}", response_model=APIResponse)
@limiter.limit("10/minute")
async def delete_thread(
    request: Request,
    thread_id: str,
    db: Session = Depends(get_db),
) -> APIResponse:  # type: ignore[type-arg]
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(404, "Thread not found.")
    db.delete(thread)
    db.commit()
    return APIResponse(data={"message": f"Thread {thread_id} deleted."})
