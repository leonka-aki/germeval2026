import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.data.load import TASKS, load_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Column name required inside each submission CSV (see shared task guidelines).
TASK_COLUMN = {"c2a": "c2a", "dbo": "dbo", "vio": "vio", "def": "def"}


def predict_task(task: str, model_dir: Path, output_dir: Path, team_name: str, run: int) -> Path:
    model_path = model_dir / task / "tfidf_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"no trained model for {task!r} at {model_path} "
            f"(run `python -m src.training.train_baseline --task {task}` first)"
        )

    pipeline = joblib.load(model_path)
    df = load_split(task, "test")
    predictions = pipeline.predict(df["text"])
    if task in ("c2a", "def"):
        # binary tasks are stored as Python bools; the spec requires the
        # literal strings TRUE/FALSE, not pandas' "True"/"False"
        predictions = ["TRUE" if p else "FALSE" for p in predictions]

    out = pd.DataFrame({"id": df["id"], TASK_COLUMN[task]: predictions})
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{team_name}{run}_{task}.csv"
    out.to_csv(out_path, sep=";", index=False)
    return out_path


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate GermEval 2026 submission CSVs from the TF-IDF baseline")
    parser.add_argument("--team", required=True, help="Team name, used in the output file names")
    parser.add_argument("--run", type=int, default=1, help="Run number (1-3), part of the file naming scheme")
    parser.add_argument(
        "--task", choices=TASKS, action="append", dest="tasks",
        help="Subtask to predict; repeat to select several. Defaults to all four.",
    )
    parser.add_argument("--models-dir", default="results/tfidf_baseline")
    parser.add_argument("--output-dir", default=None, help="Defaults to submissions/<team><run>/")
    args = parser.parse_args()

    tasks = args.tasks or list(TASKS)
    model_dir = PROJECT_ROOT / args.models_dir
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "submissions" / f"{args.team}{args.run}"

    written = []
    for task in tasks:
        try:
            path = predict_task(task, model_dir, output_dir, args.team, args.run)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        written.append(path)
        print(f"wrote {path}")

    if written:
        print(f"\n{len(written)} file(s) written to {output_dir}")
        print(f"Zip its contents into {args.team}{args.run}.zip for Codabench submission.")


if __name__ == "__main__":
    main()
