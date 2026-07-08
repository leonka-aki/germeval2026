import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data.load import load_split
from src.evaluation.explain import print_top_features
from src.evaluation.metrics import aggregate_cv_metrics, compute_metrics, report
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_pipeline(config) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(config["tfidf"]["ngram_range"]),
                    min_df=config["tfidf"]["min_df"],
                    max_features=config["tfidf"]["max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=config["logreg"]["max_iter"],
                    C=config["logreg"]["C"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def run_cv(df, config, n_folds: int, output_dir: Path) -> None:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["seed"])
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
        pipeline = build_pipeline(config)
        pipeline.fit(df["text"].iloc[train_idx], df["label"].iloc[train_idx])
        y_pred = pipeline.predict(df["text"].iloc[val_idx])
        metrics = compute_metrics(df["label"].iloc[val_idx], y_pred)
        fold_metrics.append(metrics)
        print(f"[fold {fold}/{n_folds}] macro_f1={metrics['macro_f1']:.3f}")

    summary = aggregate_cv_metrics(fold_metrics)
    print(f"\nmacro_f1: {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f} across {n_folds} folds")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved CV metrics to {output_dir / 'cv_metrics.json'}")
    print("Run again without --cv to train and save the deployable model used by predict.py.")


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline")
    parser.add_argument("--task", required=True, choices=["c2a", "dbo", "def", "vio"])
    parser.add_argument("--config", default="configs/tfidf_baseline.yaml")
    parser.add_argument(
        "--cv", type=int, default=0,
        help="If >0, run N-fold stratified CV for a more robust macro-F1 estimate, "
        "instead of the normal single train/val run (no model is saved)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])

    df = load_split(args.task, "train")
    output_dir = PROJECT_ROOT / config["output"]["directory"] / args.task

    if args.cv:
        print(f"\n=== {args.task} — TF-IDF + Logistic Regression, {args.cv}-fold CV ===")
        run_cv(df, config, args.cv, output_dir)
        return

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["label"],
        test_size=config["data"]["val_size"],
        random_state=config["seed"],
        stratify=df["label"],
    )

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    metrics = compute_metrics(y_val, y_pred)
    print(f"\n=== {args.task} — TF-IDF + Logistic Regression baseline ===")
    print(json.dumps(metrics, indent=2))
    print("\n" + report(y_val, y_pred))

    print_top_features(pipeline.named_steps["tfidf"], pipeline.named_steps["clf"], top_n=config["explain"]["top_n"])

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "tfidf_baseline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report(y_val, y_pred), encoding="utf-8")
    print(f"\nSaved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
