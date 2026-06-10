"""
Evaluation metrics with optional bootstrap confidence intervals.

All functions accept binary integer labels (0/1) and float probabilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_curve,
    auc,
    average_precision_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute the full metric set for one model on one test set.

    Parameters
    ----------
    y_true     : (N,) int array — ground-truth labels (0/1)
    y_prob     : (N,) float array — predicted probability of the positive class
    threshold  : decision threshold for converting probs to labels

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, precision, recall, f1, auc_roc, pr_auc, ap
    """
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec_curve, prec_curve)
    except ValueError:
        pr_auc = float("nan")

    try:
        ap = average_precision_score(y_true, y_prob)
    except ValueError:
        ap = float("nan")

    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision":         float(prec),
        "recall":            float(rec),
        "f1":                float(f1),
        "auc_roc":           float(roc_auc),
        "pr_auc":            float(pr_auc),
        "ap":                float(ap),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for a single metric.

    Parameters
    ----------
    y_true       : ground-truth labels
    y_prob       : predicted probabilities
    metric       : one of accuracy | balanced_accuracy | f1 | auc_roc | pr_auc
    n_bootstrap  : number of resampling iterations
    ci           : confidence level (e.g. 95 for 95%)
    random_state : RNG seed

    Returns
    -------
    {point_estimate, ci_lower, ci_upper, std}
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    def _score(yt, yp):
        m = compute_metrics(yt, yp)
        return m[metric]

    point = _score(y_true, y_prob)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(_score(y_true[idx], y_prob[idx]))

    scores = np.array(scores)
    alpha = (100 - ci) / 2
    return {
        "point_estimate": point,
        "ci_lower":       float(np.percentile(scores, alpha)),
        "ci_upper":       float(np.percentile(scores, 100 - alpha)),
        "std":            float(scores.std()),
    }


def get_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred = (y_prob >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)
