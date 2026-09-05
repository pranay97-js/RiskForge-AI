"""Unit and integration tests for RiskForge AI FastAPI backend endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app

client = TestClient(app)


def test_api_health_check():
    """Verify health endpoint responds with healthy status and metadata."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["app"] == "RiskForge AI"
    assert "RiskForge" in data["engine"]


def test_api_telemetry_metrics_and_feed():
    """Verify live telemetry returns float metrics, trendline spline, and active feed."""
    resp = client.get("/api/v1/telemetry")
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_prevented_usd" in data
    assert "verification_cost_usd" in data
    assert "active_threat_level" in data
    assert len(data["trendline"]) > 0
    assert len(data["live_feed"]) > 0
    # Check feed structure
    first_item = data["live_feed"][0]
    assert "id" in first_item
    assert "risk_score" in first_item
    assert "risk_band" in first_item


def test_api_evaluate_clean_transaction():
    """Verify evaluation of a normal low-risk transaction."""
    payload = {
        "transaction_id": "TXN_TEST_CLEAN_01",
        "amount": 1200.0,
        "payment_method": "upi",
        "card_network": "unknown",
        "merchant_category": "groceries",
        "device_type": "mobile_android",
        "ip_country": "IN",
        "is_3ds_authenticated": True,
        "customer_id": "CUST_CLEAN_100",
    }
    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["transaction_id"] == "TXN_TEST_CLEAN_01"
    assert "risk_score" in data
    assert "operational_state" in data
    assert "shap_factors" in data
    assert len(data["shap_factors"]) > 0


def test_api_evaluate_high_risk_transaction():
    """Verify evaluation of a high-value, foreign IP, non-3DS anomaly transaction."""
    payload = {
        "transaction_id": "TXN_TEST_HIGH_02",
        "amount": 95000.0,
        "payment_method": "card",
        "card_network": "visa",
        "merchant_category": "electronics",
        "device_type": "desktop_windows",
        "ip_country": "US",
        "is_3ds_authenticated": False,
        "customer_id": "CUST_SUSPECT_200",
        "is_first_transaction": True,
    }
    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_score"] > 0.3
    assert data["operational_state"] in ["REVIEW", "HIGH"]
    assert len(data["triggered_rules"]) > 0


def test_api_evidence_representment_defensible():
    """Verify building a dispute representment memo with complete evidence."""
    payload = {
        "transaction_id": "TXN_DISPUTE_001",
        "amount": 45000.0,
        "payment_proof": True,
        "three_ds_auth": True,
        "delivery_proof": True,
        "customer_communication": True,
    }
    resp = client.post("/api/v1/evidence/represent", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["readiness_score"] == 100.0
    assert data["is_defensible"] is True
    assert "Visa Compelling Evidence" in data["card_network_standard"]
    assert "DISPUTE DEFENSE" in data["dispute_memo"]


def test_api_evidence_representment_missing_pod():
    """Verify building a dispute representment memo when delivery proof is missing."""
    payload = {
        "transaction_id": "TXN_DISPUTE_002",
        "amount": 25000.0,
        "payment_proof": True,
        "three_ds_auth": True,
        "delivery_proof": False,
        "customer_communication": False,
    }
    resp = client.post("/api/v1/evidence/represent", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_defensible"] is False
    assert data["readiness_score"] == 50.0


def test_api_merchant_actions_and_history():
    """Verify execution of merchant action and tracking in audit history."""
    action_payload = {
        "transaction_id": "TXN_ACT_7712",
        "action": "Biometric Challenge",
        "analyst_notes": "Prompted biometric FaceID re-verification due to foreign IP.",
    }
    resp = client.post("/api/v1/actions/execute", json=action_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["record"]["action"] == "Biometric Challenge"

    # Retrieve history
    hist_resp = client.get("/api/v1/actions/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()["history"]
    assert len(history) > 0
    assert any(h["transaction_id"] == "TXN_ACT_7712" for h in history)


def test_api_cost_matrix_optimization():
    """Verify Bayesian cost-matrix optimization endpoint."""
    payload = {
        "investigation_cost": 50.0,
        "chargeback_penalty": 3000.0,
        "preventable_rate": 0.90,
    }
    resp = client.post("/api/v1/cost/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "optimal_threshold" in data
    assert 0.05 <= data["optimal_threshold"] <= 0.50
    assert len(data["curve_points"]) > 5
