"""Cost-sensitive threshold sweep and optimization module."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from evaluation.metrics import compute_classification_metrics
from src.decision.cost import CostEngine

logger = logging.getLogger(__name__)


def sweep_cost_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    cost_engine: CostEngine,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate business costs across a range of decision thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted chargeback probabilities.
    amounts : np.ndarray
        Transaction values.
    cost_engine : CostEngine
        Initialized cost engine instance.
    thresholds : np.ndarray, optional
        Candidate thresholds to test. Defaults to 0.05..0.95 in 0.025 steps.

    Returns
    -------
    Dict[str, Any]
        Optimal threshold, comparative metrics, and full sweep curve data.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)
    amts = np.asarray(amounts, dtype=float)

    curve_data: List[Dict[str, float]] = []
    min_cost = float("inf")
    optimal_threshold = 0.50

    for tau in thresholds:
        reviews = (y_p >= tau)
        # Calculate actual costs using ground truth
        tp = reviews & (y_t == 1)
        fp = reviews & (y_t == 0)
        fn = (~reviews) & (y_t == 1)
        tn = (~reviews) & (y_t == 0)

        potential_loss = amts + cost_engine.chargeback_penalty_fee

        # Business cost formula:
        # FP -> investigation cost
        # FN -> full loss suffered
        # TP -> investigation cost + unpreventable residual
        # TN -> zero
        investigation_costs = float(reviews.sum() * cost_engine.investigation_cost)
        missed_losses = float(potential_loss[fn].sum())
        residual_tp_losses = float((potential_loss[tp] * (1.0 - cost_engine.preventable_loss_rate)).sum())

        total_operational_cost = investigation_costs + missed_losses + residual_tp_losses

        metrics = compute_classification_metrics(y_t, y_p, threshold=float(tau))

        entry = {
            "threshold": round(float(tau), 3),
            "total_operational_cost": round(total_operational_cost, 2),
            "investigation_costs": round(investigation_costs, 2),
            "missed_chargeback_losses": round(missed_losses, 2),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "review_rate": round(float(reviews.mean()), 4),
        }
        curve_data.append(entry)

        if total_operational_cost < min_cost:
            min_cost = total_operational_cost
            optimal_threshold = float(tau)

    logger.info(
        "Threshold optimization complete: optimal cutoff=%.3f with total cost ₹%.2f",
        optimal_threshold,
        min_cost,
    )

    return {
        "optimal_threshold": optimal_threshold,
        "minimum_operational_cost": min_cost,
        "cost_curve": curve_data,
    }
