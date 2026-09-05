"""Evaluation metrics computation for imbalanced fraud/chargeback classification."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics aligned with Section 11 of the blueprint.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for class 1.
    threshold : float
        Decision cutoff for binary prediction.

    Returns
    -------
    Dict[str, Any]
        Dictionary of precision, recall, f1, pr_auc, roc_auc, brier_score, and counts.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    pr_auc_val = float(auc(recall_curve, precision_curve))

    # ROC-AUC
    try:
        roc_auc_val = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc_val = 0.5

    # Core scores
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "brier_score": brier,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_evaluated": len(y_true_arr),
        "positive_rate": float(y_true_arr.mean()),
    }
