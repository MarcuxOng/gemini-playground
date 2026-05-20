import json
import os
from sqlalchemy.orm import Session
from app.database.models import EvalDataset


def handle_seed(db: Session) -> None:
    """Seed datasets from files to DB."""
    datasets_dir = "eval/datasets"
    if not os.path.exists(datasets_dir):
        print(f"Directory {datasets_dir} does not exist.")
        return

    for filename in os.listdir(datasets_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(datasets_dir, filename)
            name = os.path.splitext(filename)[0]
            with open(filepath, encoding="utf-8") as f:
                cases = json.load(f)

            existing = db.query(EvalDataset).filter(EvalDataset.name == name).first()
            if existing:
                existing.cases = cases
                print(f"Updated dataset: {name}")
            else:
                new_dataset = EvalDataset(name=name, cases=cases)
                db.add(new_dataset)
                print(f"Created dataset: {name}")
    db.commit()
