import json
from sqlalchemy.orm import Session
from app.database.models import APIKey, EvalDataset
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

    # Get or create a dummy master API key for testing
    api_key = db.query(APIKey).first()
    if not api_key:
        api_key = APIKey(id="cli-eval-key", key="cli-eval-key")
        db.add(api_key)
        db.commit()

    print(f"Running eval dataset '{dataset.name}' ({dataset.id}) against agent '{args.agent}'...")
    res = await run_eval(db, str(dataset.id), args.agent, args.model, str(api_key.id))
    print(json.dumps(res, indent=2))
