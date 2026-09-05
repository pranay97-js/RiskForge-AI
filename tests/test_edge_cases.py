"""Comprehensive test suite covering all 12 required edge cases from Section 10.2."""

from __future__ import annotations

import pytest

from src.ai.responder import GroundedCaseResponder
from src.ai.schemas import EvidenceChecklist, GroundedCaseInput
from src.decision.fusion import DecisionFusionEngine
from src.decision.service import RiskDecisionService
from src.rules.engine import RuleEngine


@pytest.fixture
def decision_service() -> RiskDecisionService:
    return RiskDecisionService()


# Case 1: Normal low-value transaction with stable customer history
def test_case_1_normal_low_value(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C1_NORMAL",
        "timestamp": "2026-03-05 10:00:00",
        "customer_id": "CUST_00001",
        "amount": 1200.0,
        "payment_method": "upi",
        "card_network": "unknown",
        "merchant_category": "groceries",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "velocity_1h": 0,
        "velocity_ratio": 0.0,
        "amount_ratio": 0.9,
        "amount_zscore": -0.2,
    }
    res = decision_service.evaluate_transaction(txn)
    assert res["operational_decision"]["operational_state"] == "LOW"
    assert res["operational_decision"]["disagreement_detected"] is False


# Case 2: High-value transaction with strong historical deviation
def test_case_2_high_value_deviation(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C2_DEVIATION",
        "timestamp": "2026-03-05 11:00:00",
        "customer_id": "CUST_00002",
        "amount": 95000.0,
        "payment_method": "card",
        "card_network": "visa",
        "merchant_category": "electronics",
        "device_type": "desktop_windows",
        "ip_country": "IN",
        "is_3ds_authenticated": False,
        "velocity_1h": 1,
        "velocity_ratio": 1.0,
        "amount_ratio": 8.5,
        "amount_zscore": 6.2,
    }
    res = decision_service.evaluate_transaction(txn)
    assert res["operational_decision"]["operational_state"] in ["REVIEW", "HIGH"]
    assert "SPENDING_DEVIATION" in res["rule_verification"]["triggered_rule_ids"]


# Case 3: High transaction velocity case
def test_case_3_high_velocity(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C3_VELOCITY",
        "timestamp": "2026-03-05 11:30:00",
        "customer_id": "CUST_00003",
        "amount": 4000.0,
        "payment_method": "card",
        "card_network": "mastercard",
        "merchant_category": "gaming",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "velocity_1h": 6,
        "velocity_ratio": 5.2,
    }
    res = decision_service.evaluate_transaction(txn)
    assert "VELOCITY_ANOMALY" in res["rule_verification"]["triggered_rule_ids"]


# Case 4: New customer + high-value transaction
def test_case_4_new_customer_high_value(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C4_NEW_HIGH",
        "timestamp": "2026-03-05 12:00:00",
        "customer_id": "CUST_NEW_9999",
        "amount": 45000.0,
        "payment_method": "card",
        "card_network": "visa",
        "merchant_category": "electronics",
        "device_type": "mobile_ios",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "is_first_transaction": True,
    }
    res = decision_service.evaluate_transaction(txn)
    assert "NEW_ACCOUNT_HIGH_VALUE" in res["rule_verification"]["triggered_rule_ids"]


# Case 5: Transaction with prior dispute history
def test_case_5_dispute_history(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C5_DISPUTE",
        "timestamp": "2026-03-05 12:30:00",
        "customer_id": "CUST_FLAGGED_11",
        "amount": 5000.0,
        "payment_method": "card",
        "card_network": "visa",
        "merchant_category": "digital_goods",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "prior_dispute_count": 2,
    }
    res = decision_service.evaluate_transaction(txn)
    assert "DISPUTE_HISTORY" in res["rule_verification"]["triggered_rule_ids"]
    assert res["operational_decision"]["operational_state"] in ["REVIEW", "HIGH"]


# Case 6: ML-high / rules-low disagreement case
def test_case_6_ml_high_rules_low_disagreement():
    fusion = DecisionFusionEngine()
    decision = fusion.fuse(ml_score=0.88, rule_signal="Low", data_quality="Good", expected_loss=10000.0, priority_score=100.0)
    assert decision.operational_state == "REVIEW"
    assert decision.disagreement_detected is True
    assert "ML_LEAD" in decision.reason_code


# Case 7: ML-low / rules-high disagreement case
def test_case_7_ml_low_rules_high_disagreement():
    fusion = DecisionFusionEngine()
    decision = fusion.fuse(ml_score=0.12, rule_signal="High", data_quality="Good", expected_loss=2000.0, priority_score=20.0)
    assert decision.operational_state == "REVIEW"
    assert decision.disagreement_detected is True
    assert "RULE_LEAD" in decision.reason_code


# Case 8: Missing customer-history data (cold start)
def test_case_8_cold_start_customer(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C8_COLD",
        "timestamp": "2026-03-05 13:00:00",
        "customer_id": "CUST_COLD_01",
        "amount": 3500.0,
        "payment_method": "upi",
        "card_network": "unknown",
        "merchant_category": "fashion",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "is_first_transaction": True,
        "customer_prior_txn_count": 0,
    }
    res = decision_service.evaluate_transaction(txn)
    assert res["operational_decision"]["operational_state"] in ["LOW", "REVIEW"]


# Case 9: Extreme amount / outlier input
def test_case_9_extreme_outlier_amount(decision_service: RiskDecisionService):
    txn = {
        "transaction_id": "TXN_C9_EXTREME",
        "timestamp": "2026-03-05 13:30:00",
        "customer_id": "CUST_00001",
        "amount": 2500000.0,  # ₹25 Lakhs outlier
        "payment_method": "card",
        "card_network": "amex",
        "merchant_category": "electronics",
        "device_type": "desktop_mac",
        "ip_country": "US",
        "is_3ds_authenticated": False,
        "amount_ratio": 50.0,
        "amount_zscore": 25.0,
    }
    res = decision_service.evaluate_transaction(txn)
    assert res["operational_decision"]["operational_state"] == "HIGH"
    assert res["financial_exposure"]["expected_loss"] > 100000.0


# Case 10: Malformed or missing transaction ID
def test_case_10_missing_txn_id(decision_service: RiskDecisionService):
    txn = {
        "timestamp": "2026-03-05 14:00:00",
        "customer_id": "CUST_00005",
        "amount": 2500.0,
        "payment_method": "upi",
        "card_network": "unknown",
        "merchant_category": "groceries",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
    }
    res = decision_service.evaluate_transaction(txn)
    # Service should handle safely without crashing
    assert "transaction_id" in res
    assert res["transaction_id"] != ""


# Case 11: LLM unavailable / API failure
def test_case_11_llm_api_failure():
    # Force mock/fallback mode to verify deterministic fallback
    responder = GroundedCaseResponder(provider="invalid_provider")
    inp = GroundedCaseInput(
        transaction_id="TXN_C11_FALLBACK",
        risk_score=0.85,
        risk_level="HIGH",
        rules_triggered=["VELOCITY_ANOMALY"],
        top_model_factors=["1-Hour Order Frequency"],
        evidence=EvidenceChecklist(payment_proof=True, three_ds_auth=True, delivery_proof=True, customer_communication=False),
        expected_loss=15000.0,
        decision="HIGH",
        amount=12000.0,
    )
    output = responder.generate_case_response(inp)
    assert output is not None
    assert output.transaction_id == "TXN_C11_FALLBACK"
    assert output.is_defense_ready is True


# Case 12: Conflicting or incomplete evidence
def test_case_12_conflicting_incomplete_evidence():
    responder = GroundedCaseResponder()
    inp = GroundedCaseInput(
        transaction_id="TXN_C12_CONFLICT",
        risk_score=0.72,
        risk_level="REVIEW",
        rules_triggered=["PAYMENT_CHANGE"],
        top_model_factors=["Cross Border IP"],
        # Conflicting evidence: 3DS passed, but courier POD is missing
        evidence=EvidenceChecklist(payment_proof=True, three_ds_auth=True, delivery_proof=False, customer_communication=False),
        expected_loss=8000.0,
        decision="REVIEW",
        amount=7500.0,
    )
    output = responder.generate_case_response(inp)
    assert output.is_defense_ready is False
    assert output.evidence_completeness_score == 50.0
    assert "[DISPUTE DEFENSE NOT READY]" in output.dispute_response_draft
    assert "Signed Delivery Proof" in output.dispute_response_draft
