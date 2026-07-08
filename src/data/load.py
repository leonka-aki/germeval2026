from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TASKS = ("c2a", "dbo", "def", "vio")


def load_split(task: str, split: str) -> pd.DataFrame:
    """Load one GermEval 2026 subtask split as a DataFrame.

    task: one of "c2a", "dbo", "def", "vio"
    split: one of "trial", "train", "test"

    Returns columns "id", "text", and (for "trial"/"train") "label".
    File naming is inconsistent across subtasks (e.g. "c2a_train_26.csv"
    vs "def_train.csv"), so the file is located with a glob instead of
    a fixed name.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}, expected one of {TASKS}")

    task_dir = RAW_DIR / task
    pattern = f"{task}_trial.csv" if split == "trial" else f"{task}_{split}*.csv"

    matches = sorted(task_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {task_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple files matching {pattern!r} in {task_dir}: {matches}")

    df = pd.read_csv(matches[0], sep=";")
    df = df.rename(columns={df.columns[1]: "text"})
    if df.shape[1] == 3:
        df = df.rename(columns={df.columns[2]: "label"})
    return df
