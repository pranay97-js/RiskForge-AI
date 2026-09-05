"""Grounded AI case explanation and dispute response generator."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from src.ai.prompts import GROUNDED_SYSTEM_PROMPT, build_user_case_prompt
from src.ai.schemas import EvidenceChecklist, GroundedCaseInput, GroundedCaseOutput

logger = logging.getLogger(__name__)


class GroundedCaseResponder:
    """Generates grounded, auditable case explanations and dispute response drafts."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "mock").lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def generate_case_response(self, case_input: GroundedCaseInput | Dict[str, Any]) -> GroundedCaseOutput:
        """Generate verified case explanation and dispute draft."""
        if isinstance(case_input, dict):
            # Parse into validated Pydantic model
            if "evidence" in case_input and isinstance(case_input["evidence"], dict):
                case_input["evidence"] = EvidenceChecklist(**case_input["evidence"])
            validated_input = GroundedCaseInput(**case_input)
        else:
            validated_input = case_input

        # Check if external LLM API is requested and available
        if self.provider == "openai" and self.openai_api_key:
            try:
                return self._call_openai_api(validated_input)
            except Exception as e:
                logger.warning("OpenAI API call failed (%s); falling back to deterministic generator.", e)

        elif self.provider == "gemini":
            logger.info("Gemini provider selected but not yet implemented; using deterministic generator.")

        # Default: High-reliability deterministic grounded generator
        return self._generate_deterministic_response(validated_input)

    def _generate_deterministic_response(self, inp: GroundedCaseInput) -> GroundedCaseOutput:
        """Deterministic, grounded synthesis strictly conforming to supplied input JSON."""
        # 1. Synthesize specific flag reasons from SHAP factors and rules
        reasons = []
        for rule in inp.rules_triggered:
            rule_clean = rule.replace("_", " ").title()
            reasons.append(f"Deterministic Rule Triggered: {rule_clean}")

        for factor in inp.top_model_factors:
            factor_clean = factor.replace("_", " ").title()
            reasons.append(f"ML Statistical Factor: {factor_clean} exhibited elevated deviation")

        if not reasons:
            reasons.append("Transaction matches normal baseline spending; no anomalous deviations detected.")

        # 2. Evidence status map & score
        ev = inp.evidence
        completeness = ev.completeness_percentage()
        ev_status = {
            "Payment Receipt / Gateway ID": "Verified" if ev.payment_proof else "Missing",
            "3D-Secure Authentication Log": "Passed (OTP/Biometric)" if ev.three_ds_auth else "Missing / Unverified",
            "Signed Delivery Proof (POD)": "Verified (Signed)" if ev.delivery_proof else "Missing",
            "Customer Communication": "Available" if ev.customer_communication else "Missing",
        }

        # 3. Executive case summary
        if inp.decision == "HIGH":
            summary = (
                f"Transaction {inp.transaction_id} flagged as HIGH RISK (Score: {inp.risk_score:.2f}) "
                f"with estimated exposure of ₹{inp.expected_loss:,.2f}. Corroborated by {len(inp.rules_triggered)} "
                f"deterministic rule(s) and statistical anomaly detection."
            )
        elif inp.decision == "REVIEW":
            summary = (
                f"Transaction {inp.transaction_id} routed to MANUAL REVIEW (Score: {inp.risk_score:.2f}). "
                f"Expected exposure is ₹{inp.expected_loss:,.2f}; operational verification is required before settlement."
            )
        else:
            summary = (
                f"Transaction {inp.transaction_id} cleared as LOW RISK (Score: {inp.risk_score:.2f}). "
                f"Features align with established customer baseline and security protocols."
            )

        # 4. Recommended Action
        if inp.decision == "HIGH":
            action = "Immediately hold order fulfillment, inspect courier dispatch, and prepare dispute defense package."
        elif inp.decision == "REVIEW":
            action = "Contact merchant operations analyst to cross-verify customer identity and billing address."
        else:
            action = "Auto-approve transaction for settlement under standard monitoring policy."

        # 5. Formal Dispute Defense Draft (Generated ONLY when evidence is defensible)
        defense_ready = ev.is_dispute_defensible()
        if defense_ready:
            amt_str = f"₹{inp.amount:,.2f}" if inp.amount is not None else f"₹{inp.expected_loss:,.2f}"
            from datetime import date
            draft = (
                f"DISPUTE DEFENSE MEMORANDUM\n"
                f"To: Chargeback & Dispute Operations Review Committee\n"
                f"From: Merchant Fraud Prevention Team (via Razorpay RiskForge AI)\n"
                f"Date: {date.today().isoformat()}\n"
                f"Subject: Formal Rebuttal & Compelling Evidence for Disputed Transaction {inp.transaction_id}\n\n"
                f"Dear Dispute Operations Team,\n\n"
                f"We are formally submitting compelling evidence to contest the chargeback on transaction "
                f"ID {inp.transaction_id} (Disputed Amount: {amt_str}).\n\n"
                f"SUMMARY OF VERIFIED COMPELLING EVIDENCE:\n"
                f"1. 3D-Secure Authentication: The transaction completed strong customer authentication (3DS OTP) "
                f"at the time of purchase, fulfilling card network liability-shift criteria.\n"
                f"2. Validated Gateway Receipt: Payment successfully settled through Razorpay gateway with full authorization code.\n"
                f"3. Proof of Delivery: Physical goods were dispatched to the cardholder's verified billing address "
                f"and signed for upon receipt (Tracking POD confirmed).\n\n"
                f"Under network chargeback dispute guidelines, this documented evidence demonstrates that the legitimate "
                f"cardholder authorized and received the purchase. We respectfully request immediate reversal and credit "
                f"of the disputed funds to the merchant account.\n\n"
                f"Sincerely,\n"
                f"Merchant Operations Team"
            )
        else:
            missing_items = [k for k, v in ev_status.items() if "Missing" in v]
            draft = (
                f"[DISPUTE DEFENSE NOT READY]\n"
                f"Evidence completeness is {completeness}%. A formal rebuttal letter cannot be safely submitted "
                f"until the following missing documents are retrieved:\n"
                + "\n".join([f"  • {item}" for item in missing_items])
                + "\n\nRecommendation: Request proof of delivery and communication logs before submitting dispute response."
            )

        return GroundedCaseOutput(
            transaction_id=inp.transaction_id,
            case_summary=summary,
            flag_reasons=reasons,
            evidence_status=ev_status,
            evidence_completeness_score=completeness,
            recommended_action=action,
            dispute_response_draft=draft,
            is_defense_ready=defense_ready,
            model_version="RiskForge-Grounded-Deterministic-v1.0",
        )

    def _call_openai_api(self, inp: GroundedCaseInput) -> GroundedCaseOutput:
        """Call OpenAI API using structured JSON output mode (if key provided)."""
        import openai  # optional dependency
        client = openai.OpenAI(api_key=self.openai_api_key)

        user_prompt = build_user_case_prompt(inp.model_dump())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content
        import json
        data = json.loads(content)
        return GroundedCaseOutput(**data)
