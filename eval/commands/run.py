import argparse
import json
from sqlalchemy.orm import Session
from app.database.models import EvalDataset
from app.services.evals import run_eval


async def handle_run(args: argparse.Namespace, db: Session) -> None:
    """Run an evaluation."""
    # Find dataset by ID or Name
    dataset: EvalDataset | None = db.query(EvalDataset).filter(
        (EvalDataset.id == args.dataset) | (EvalDataset.name == args.dataset)
    ).first()
    
    if not dataset:
        print(f"Dataset '{args.dataset}' not found.")
        return

    print(f"Running eval dataset '{dataset.name}' ({dataset.id}) against agent '{args.agent}'...")
    # Use the "master" ID (not the raw secret) so run_eval gets an identity, not a credential.
    res = await run_eval(db, str(dataset.id), args.agent, args.model, "master")
    print(json.dumps(res, indent=2))
