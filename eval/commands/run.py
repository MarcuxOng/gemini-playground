import json
from sqlalchemy.orm import Session
from app.config import settings
from app.database.models import EvalDataset
from app.services.evals import run_eval


async def handle_run(args, db: Session) -> None:
    """Run an evaluation."""
    # Find dataset by ID or Name
    dataset: EvalDataset | None = db.query(EvalDataset).filter(
        (EvalDataset.id == args.dataset) | (EvalDataset.name == args.dataset)
    ).first()
    
    if not dataset:
        print(f"Dataset '{args.dataset}' not found.")
        return

    print(f"Running eval dataset '{dataset.name}' ({dataset.id}) against agent '{args.agent}'...")
    res = await run_eval(db, str(dataset.id), args.agent, args.model, settings.master_api_key)
    print(json.dumps(res, indent=2))
