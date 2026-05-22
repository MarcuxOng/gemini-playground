import json
from pathlib import Path
from sqlalchemy.orm import Session
from app.database.models import EvalDataset


def handle_seed(db: Session) -> None:
    """Seed datasets from files to DB."""
    datasets_dir = Path(__file__).resolve().parent.parent / "datasets"
    if not datasets_dir.exists():
        print(f"Directory {datasets_dir} does not exist.")
        return

    for filepath in datasets_dir.iterdir():
        if filepath.suffix != ".json":
            continue
        name = filepath.stem
        try:
            with open(filepath, encoding="utf-8") as f:
                cases = json.load(f)

            if not isinstance(cases, list):
                print(f"Skipping {name}: top-level value must be a list of cases")
                continue

            for i, case in enumerate(cases):
                if not isinstance(case, dict) or "input" not in case or "expected" not in case:
                    print(f"Skipping {name}: case {i} missing required keys 'input' or 'expected'")
                    break
            else:
                existing = db.query(EvalDataset).filter(EvalDataset.name == name).first()
                if existing:
                    existing.cases = cases  # type: ignore[assignment]
                    print(f"Updated dataset: {name}")
                else:
                    db.add(EvalDataset(name=name, cases=cases))
                    print(f"Created dataset: {name}")
                db.commit()

        except Exception as e:
            db.rollback()
            print(f"Error processing {name}: {e}")
