"""Unit tests for Grounded AI schemas, prompts, and responder."""

from __future__ import annotations

import pytest

from src.ai.responder import GroundedCaseResponder
from src.ai.schemas import EvidenceChecklist, GroundedCaseInput, GroundedCaseOutput


def test_evidence_checklist_completeness():
    # 2 out of 4 items true
    ev = EvidenceChecklist(payment_proof=True, three_ds_auth=True, delivery_proof=False, customer_communication=False)
    assert ev.completeness_percentage() == 50.0
    assert not ev.is_dispute_defensible()

    # 3 critical items true (payment, 3ds, delivery)
    ev_defensible = EvidenceChecklist(payment_proof=True, three_ds_auth=True, delivery_proof=True, customer_communication=False)
    assert ev_defensible.completeness_percentage() == 75.0
    assert ev_defensible.is_dispute_defensible()


def test_grounded_case_responder_defensible():
    responder = GroundedCaseResponder()

    inp = GroundedCaseInput(
        transaction_id="TXN_TEST_99",
        risk_score=0.92,
        risk_level="HIGH",
        rules_triggered=["VELOCITY_ANOMALY", "SPENDING_DEVIATION"],
        top_model_factors=["1-Hour Order Frequency", "Spending Z-Score Deviation"],
        evidence=EvidenceChecklist(
            payment_proof=True,
            three_ds_auth=True,
            delivery_proof=True,
            customer_communication=False,
        ),
        expected_loss=85000.0,
        decision="HIGH",
        amount=82000.0,
    )

    output = responder.generate_case_response(inp)

    assert isinstance(output, GroundedCaseOutput)
    assert output.transaction_id == "TXN_TEST_99"
    assert output.is_defense_ready is True
    assert "DISPUTE DEFENSE MEMORANDUM" in output.dispute_response_draft
    assert "3D-Secure" in output.dispute_response_draft
    assert len(output.flag_reasons) >= 2


def test_grounded_case_responder_missing_evidence():
    responder = GroundedCaseResponder()

    inp = GroundedCaseInput(
        transaction_id="TXN_MISSING_01",
        risk_score=0.75,
        risk_level="HIGH",
        rules_triggered=["SPENDING_DEVIATION"],
        top_model_factors=["Spending Z-Score Deviation"],
        evidence=EvidenceChecklist(
            payment_proof=True,
            three_ds_auth=False,
            delivery_proof=False,
            customer_communication=False,
        ),
        expected_loss=40000.0,
        decision="REVIEW",
        amount=38000.0,
    )

    output = responder.generate_case_response(inp)

    assert output.is_defense_ready is False
    assert output.evidence_completeness_score == 25.0
    assert "[DISPUTE DEFENSE NOT READY]" in output.dispute_response_draft
    assert "missing documents" in output.dispute_response_draft.lower()

