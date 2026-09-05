"""Confusion matrix calculation, formatting, and false positive analysis."""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import confusion_matrix


def generate_confusion_matrix_report(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> Dict[str, int | float]:
    """Generate detailed breakdown of confusion matrix and rate metrics."""
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fdr = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "false_discovery_rate": fdr,
    }


def format_confusion_matrix_ascii(cm_dict: Dict[str, int | float]) -> str:
    """Return formatted ASCII table of confusion matrix for experiment logging."""
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    report = (
        "--------------------------------------------------\n"
        "                PREDICTED: 0        PREDICTED: 1  \n"
        f"ACTUAL: 0       TN: {tn:<14}  FP: {fp:<14}\n"
        f"ACTUAL: 1       FN: {fn:<14}  TP: {tp:<14}\n"
        "--------------------------------------------------\n"
        f"False Positive Rate (FPR): {cm_dict['false_positive_rate']*100:.2f}%\n"
        f"False Discovery Rate (FDR): {cm_dict['false_discovery_rate']*100:.2f}%\n"
    )
    return report
