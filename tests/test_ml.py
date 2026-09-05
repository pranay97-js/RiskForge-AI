"""Tests for ML metrics, calibration, and prediction pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.confusion_matrix import format_confusion_matrix_ascii, generate_confusion_matrix_report
from evaluation.metrics import compute_classification_metrics
from src.data.loader import generate_benchmark_transactions
from src.features.pipeline import RiskFeaturePipeline
from src.ml.predict import RiskPredictor


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] > 0.9
    assert metrics["roc_auc"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0


def test_confusion_matrix_report():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cm = generate_confusion_matrix_report(y_true, y_pred)
    assert cm["true_negatives"] == 1
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 1

    ascii_str = format_confusion_matrix_ascii(cm)
    assert "False Positive Rate" in ascii_str


def test_risk_predictor_saved_artifacts():
    import pandas as pd
    predictor = RiskPredictor()

    sample_txn = {
        "transaction_id": "TXN_TEST_001",
        "timestamp": "2026-03-05 12:00:00",
        "customer_id": "CUST_00001",
        "amount": 15000.0,
        "payment_method": "card",
        "card_network": "visa",
        "merchant_category": "electronics",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "is_chargeback": 0,
    }

    result = predictor.predict_single(sample_txn)
    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_level"] in ["LOW", "REVIEW", "HIGH"]
    assert result["transaction_id"] == "TXN_TEST_001"

