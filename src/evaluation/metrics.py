from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


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
