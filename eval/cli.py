import argparse
from app.database.db import SessionLocal
from eval.commands.seed import handle_seed
from eval.commands.run import handle_run


async def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Harness CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Seed command
    subparsers.add_parser("seed", help="Seed datasets from files to DB")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    run_parser.add_argument("dataset", help="Dataset ID or Name")
    run_parser.add_argument("agent", help="Agent preset (research, coder, analyst, knowledge)")
    run_parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use")

    args = parser.parse_args()
    db = SessionLocal()

    try:
        if args.command == "seed":
            handle_seed(db)
        elif args.command == "run":
            await handle_run(args, db)
    finally:
        db.close()
