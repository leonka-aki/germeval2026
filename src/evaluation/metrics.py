import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def aggregate_cv_metrics(fold_metrics: list) -> dict:
    """Mean/std of each metric across CV folds, plus the raw per-fold values
    (useful since rare classes have single-digit support per fold and a
    single split's macro-F1 is otherwise noisy and easy to over-read)."""
    summary = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["folds"] = fold_metrics
    return summary


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """Macro-averaged metrics, appropriate here since all four GermEval
    subtasks are heavily class-imbalanced (the harmful classes are rare)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }


def report(y_true, y_pred, labels=None) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)
