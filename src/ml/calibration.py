"""Probability calibration routines for risk scoring."""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Calibrate model predicted probabilities using validation data.

    Parameters
    ----------
    model : fitted estimator
        Fitted base classifier.
    X_val : np.ndarray
        Validation feature matrix.
    y_val : np.ndarray
        Validation ground-truth labels.
    method : str
        'sigmoid' (Platt scaling) or 'isotonic'.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model wrapper.
    """
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)

    # Check Brier score improvement
    orig_prob = model.predict_proba(X_val)[:, 1]
    cal_prob = calibrated.predict_proba(X_val)[:, 1]

    orig_brier = brier_score_loss(y_val, orig_prob)
    cal_brier = brier_score_loss(y_val, cal_prob)

    logger.info(
        "Calibration (%s) complete: Brier score %.4f -> %.4f",
        method,
        orig_brier,
        cal_brier,
    )
    return calibrated
