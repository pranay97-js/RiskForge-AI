"""Unit tests for decision fusion engine and end-to-end RiskDecisionService."""

from __future__ import annotations

import pytest

from src.decision.fusion import DecisionFusionEngine
from src.decision.service import RiskDecisionService


def test_decision_fusion_convergent_high():
    fusion = DecisionFusionEngine()
    decision = fusion.fuse(
        ml_score=0.92,
        rule_signal="High",
        data_quality="Good",
        expected_loss=15000.0,
        priority_score=150.0,
    )
    assert decision.operational_state == "HIGH"
    assert decision.confidence >= 0.90
    assert not decision.disagreement_detected


def test_decision_fusion_convergent_low():
    fusion = DecisionFusionEngine()
    decision = fusion.fuse(
        ml_score=0.08,
        rule_signal="Low",
        data_quality="Good",
        expected_loss=150.0,
        priority_score=1.0,
    )
    assert decision.operational_state == "LOW"
    assert decision.confidence >= 0.90
    assert not decision.disagreement_detected


def test_decision_fusion_disagreement_cases():
    fusion = DecisionFusionEngine()

    # ML High, Rule Low
    d1 = fusion.fuse(0.85, "Low", "Good", 5000.0, 50.0)
    assert d1.operational_state == "REVIEW"
    assert d1.disagreement_detected is True

    # ML Low, Rule High
    d2 = fusion.fuse(0.10, "High", "Good", 3000.0, 30.0)
    assert d2.operational_state == "REVIEW"
    assert d2.disagreement_detected is True


def test_decision_fusion_poor_data_quality_override():
    fusion = DecisionFusionEngine()
    # Even if ML is 0.99 and Rule is High, Poor data quality MUST force REVIEW
    d = fusion.fuse(0.99, "High", "Poor", 80000.0, 800.0)
    assert d.operational_state == "REVIEW"
    assert "POOR" in d.reason_code


def test_decision_service_end_to_end():
    service = RiskDecisionService()

    txn = {
        "transaction_id": "TXN_LIVE_999",
        "timestamp": "2026-03-05 14:30:00",
        "customer_id": "CUST_00042",
        "amount": 75000.0,
        "payment_method": "card",
        "card_network": "visa",
        "merchant_category": "electronics",
        "device_type": "mobile_android",
        "ip_country": "US",
        "is_3ds_authenticated": False,
        "is_chargeback": 0,
        "velocity_1h": 4,
        "velocity_ratio": 4.5,
        "amount_ratio": 6.2,
        "amount_zscore": 5.1,
    }

    result = service.evaluate_transaction(txn)

    assert result["transaction_id"] == "TXN_LIVE_999"
    assert "operational_decision" in result
    assert result["operational_decision"]["operational_state"] in ["LOW", "REVIEW", "HIGH"]
    assert "ml_risk" in result
    assert "rule_verification" in result
    assert "financial_exposure" in result
    assert "explainability" in result
