import numpy as np


def top_features(vectorizer, classifier, top_n=15) -> dict:
    """Highest/lowest-weighted TF-IDF features per class of a fitted linear
    classifier (LogisticRegression/LinearSVC) — a simple, inspectable
    explanation of what the model learned, no extra XAI library needed.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = classifier.coef_  # shape: (n_classes_or_1, n_features)
    classes = classifier.classes_

    result = {}
    if coef.shape[0] == 1:
        # binary case: sklearn stores one coefficient row; positive weights
        # push toward classes_[1], negative weights toward classes_[0]
        order = np.argsort(coef[0])
        top_pos = order[::-1][:top_n]
        top_neg = order[:top_n]
        result[classes[1]] = list(zip(feature_names[top_pos], coef[0][top_pos]))
        result[classes[0]] = list(zip(feature_names[top_neg], coef[0][top_neg]))
    else:
        for i, cls in enumerate(classes):
            order = np.argsort(coef[i])[::-1][:top_n]
            result[cls] = list(zip(feature_names[order], coef[i][order]))
    return result


def print_top_features(vectorizer, classifier, top_n=15) -> None:
    for cls, feats in top_features(vectorizer, classifier, top_n).items():
        print(f"\nTop features for class {cls!r}:")
        for name, weight in feats:
            print(f"  {weight:+.3f}  {name}")
