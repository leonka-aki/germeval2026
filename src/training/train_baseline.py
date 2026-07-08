import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load import load_split
from src.evaluation.explain import print_top_features
from src.evaluation.metrics import compute_metrics, report
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline")
    parser.add_argument("--task", required=True, choices=["c2a", "dbo", "def", "vio"])
    parser.add_argument("--config", default="configs/tfidf_baseline.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])

    df = load_split(args.task, "train")
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        df["label"],
        test_size=config["data"]["val_size"],
        random_state=config["seed"],
        stratify=df["label"],
    )

    pipeline = Pipeline(
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

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    metrics = compute_metrics(y_val, y_pred)
    print(f"\n=== {args.task} — TF-IDF + Logistic Regression baseline ===")
    print(json.dumps(metrics, indent=2))
    print("\n" + report(y_val, y_pred))

    print_top_features(pipeline.named_steps["tfidf"], pipeline.named_steps["clf"], top_n=config["explain"]["top_n"])

    output_dir = PROJECT_ROOT / config["output"]["directory"] / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "tfidf_baseline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report(y_val, y_pred), encoding="utf-8")
    print(f"\nSaved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
