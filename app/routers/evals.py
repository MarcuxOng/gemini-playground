from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import APIKey, EvalDataset, EvalRun
from app.services.evals import run_eval
from app.utils.auth import verify_api_key
from app.utils.limiter import limiter
from app.utils.response import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/evals", tags=["Evals"], dependencies=[Depends(verify_api_key)])


class DatasetCreate(BaseModel):
    name: str
    cases: list[dict[str, Any]]  # [{input, expected}]


class EvalRunRequest(BaseModel):
    dataset_id: str
    agent_id_or_preset: str
    model: str


@router.post("/datasets", response_model=APIResponse)
@limiter.limit("10/minute")
async def create_dataset(
    request: Request, body: DatasetCreate, db: Session = Depends(get_db)
) -> APIResponse:  # type: ignore[type-arg]
    existing = db.query(EvalDataset).filter(EvalDataset.name == body.name).first()
    if existing:
        raise HTTPException(400, f"Dataset with name '{body.name}' already exists.")

    dataset = EvalDataset(name=body.name, cases=body.cases)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return APIResponse(data={"id": dataset.id, "name": dataset.name})


@router.get("/datasets", response_model=APIResponse)
async def list_datasets(db: Session = Depends(get_db)) -> APIResponse:  # type: ignore[type-arg]
    datasets = db.query(EvalDataset).all()
    return APIResponse(data=[{"id": d.id, "name": d.name} for d in datasets])


@router.post("/run", response_model=APIResponse)
@limiter.limit("5/minute")
async def start_eval(
    request: Request,
    body: EvalRunRequest,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(verify_api_key),
) -> APIResponse:  # type: ignore[type-arg]
    logger.info(
        f"Starting eval run for dataset {body.dataset_id} with agent {body.agent_id_or_preset}"
    )
    result = await run_eval(
        db, body.dataset_id, body.agent_id_or_preset, body.model, str(api_key.id)
    )
    return APIResponse(data=result)


@router.get("/runs/{run_id}", response_model=APIResponse)
async def get_eval_run(run_id: str, db: Session = Depends(get_db)) -> APIResponse:  # type: ignore[type-arg]
    eval_run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not eval_run:
        raise HTTPException(404, "Eval run not found.")
    return APIResponse(
        data={
            "id": eval_run.id,
            "dataset_id": eval_run.dataset_id,
            "agent_id": eval_run.agent_id,
            "metrics": eval_run.metrics,
            "created_at": eval_run.created_at,
        }
    )
