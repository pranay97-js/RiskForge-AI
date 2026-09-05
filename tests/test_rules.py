"""Unit tests for deterministic rules, registry, and rule engine."""

from __future__ import annotations

import pytest

from src.rules.engine import RuleEngine
from src.rules.registry import RuleRegistry
from src.rules.rules import (
    DataInsufficientRule,
    DisputeHistoryRule,
    NewAccountHighValueRule,
    PaymentChangeRule,
    RuleSeverity,
    SpendingDeviationRule,
    VelocityAnomalyRule,
)


def test_velocity_anomaly_rule():
    rule = VelocityAnomalyRule(velocity_1h_threshold=3, velocity_ratio_threshold=3.0)

    # Normal transaction
    res_normal = rule.evaluate({"velocity_1h": 1, "velocity_ratio": 1.2})
    assert not res_normal.triggered

    # Triggered by count
    res_count = rule.evaluate({"velocity_1h": 4, "velocity_ratio": 1.0})
    assert res_count.triggered
    assert res_count.severity == RuleSeverity.HIGH

    # Triggered by ratio
    res_ratio = rule.evaluate({"velocity_1h": 2, "velocity_ratio": 4.5})
    assert res_ratio.triggered


def test_spending_deviation_rule():
    rule = SpendingDeviationRule(amount_ratio_threshold=3.5, min_amount_floor=5000.0)

    # Normal spending
    res_normal = rule.evaluate({"amount": 2000.0, "amount_ratio": 1.1, "amount_zscore": 0.5})
    assert not res_normal.triggered

    # High ratio but small amount below floor -> cleared
    res_small = rule.evaluate({"amount": 300.0, "amount_ratio": 5.0, "amount_zscore": 1.0})
    assert not res_small.triggered

    # High ratio and above floor -> triggered
    res_spike = rule.evaluate({"amount": 25000.0, "amount_ratio": 4.2, "amount_zscore": 4.0})
    assert res_spike.triggered


def test_dispute_history_rule():
    rule = DisputeHistoryRule()

    assert not rule.evaluate({"prior_dispute_count": 0}).triggered
    assert rule.evaluate({"prior_dispute_count": 2}).triggered
    assert rule.evaluate({"has_prior_chargeback": True}).triggered


def test_new_account_high_value_rule():
    rule = NewAccountHighValueRule(high_value_threshold=20000.0)

    # Old customer high value -> cleared
    assert not rule.evaluate({"is_first_transaction": False, "amount": 50000.0}).triggered

    # New customer low value -> cleared
    assert not rule.evaluate({"is_first_transaction": True, "amount": 1500.0}).triggered

    # New customer high value -> triggered
    assert rule.evaluate({"is_first_transaction": True, "amount": 35000.0}).triggered


def test_payment_change_rule():
    rule = PaymentChangeRule()

    # 3DS passed domestic -> cleared
    assert not rule.evaluate({"is_3ds_authenticated": True, "ip_country": "IN", "amount": 50000.0}).triggered

    # Non-3DS foreign IP -> triggered
    assert rule.evaluate({"is_3ds_authenticated": False, "ip_country": "US", "amount": 5000.0}).triggered


def test_data_insufficient_rule():
    rule = DataInsufficientRule()

    # Complete valid data
    assert not rule.evaluate({"amount": 500.0, "customer_id": "CUST_1", "payment_method": "upi"}).triggered

    # Missing amount
    assert rule.evaluate({"amount": 0.0, "customer_id": "CUST_1", "payment_method": "upi"}).triggered

    # Missing customer_id
    assert rule.evaluate({"amount": 500.0, "customer_id": "", "payment_method": "upi"}).triggered


def test_rule_engine_execution():
    engine = RuleEngine()
    
    # Anomaly payload
    anomaly_txn = {
        "amount": 45000.0,
        "amount_ratio": 5.0,
        "amount_zscore": 4.5,
        "velocity_1h": 5,
        "velocity_ratio": 4.0,
        "customer_id": "CUST_888",
        "payment_method": "card",
        "is_3ds_authenticated": False,
        "ip_country": "US",
        "is_first_transaction": False,
    }

    output = engine.evaluate(anomaly_txn)
    assert len(output.triggered_rules) >= 3
    assert output.rule_severity_score >= 0.50
    assert output.rule_signal == "High"
    assert output.data_quality == "Good"
