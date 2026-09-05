"""Decision fusion engine: deterministic synthesis of ML, rules, data quality, and cost.

Corresponds to Section 4.6 of the blueprint:
No hidden LLM calls; completely deterministic and auditable operational decisions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class FusedDecision:
    """The final operational decision emitted by the decision fusion engine."""

    def __init__(
        self,
        operational_state: str,
        recommended_action: str,
        confidence: float,
        reason_code: str,
        ml_band: str,
        rule_signal: str,
        data_quality: str,
        expected_loss: float,
        priority_score: float,
        disagreement_detected: bool,
    ):
        self.operational_state = operational_state  # "LOW", "REVIEW", "HIGH"
        self.recommended_action = recommended_action
        self.confidence = confidence
        self.reason_code = reason_code
        self.ml_band = ml_band
        self.rule_signal = rule_signal
        self.data_quality = data_quality
        self.expected_loss = expected_loss
        self.priority_score = priority_score
        self.disagreement_detected = disagreement_detected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operational_state": self.operational_state,
            "recommended_action": self.recommended_action,
            "confidence": round(self.confidence, 4),
            "reason_code": self.reason_code,
            "ml_band": self.ml_band,
            "rule_signal": self.rule_signal,
            "data_quality": self.data_quality,
            "expected_loss": round(self.expected_loss, 2),
            "priority_score": round(self.priority_score, 2),
            "disagreement_detected": self.disagreement_detected,
        }


class DecisionFusionEngine:
    """Combines ML risk probability, deterministic rule checks, data quality, and financial exposure."""

    def __init__(
        self,
        high_risk_threshold: float = 0.65,
        low_risk_threshold: float = 0.25,
        review_loss_floor: float = 2500.0,
    ):
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold
        self.review_loss_floor = review_loss_floor

    def _determine_ml_band(self, ml_score: float) -> str:
        if ml_score >= self.high_risk_threshold:
            return "High"
        elif ml_score >= self.low_risk_threshold:
            return "Medium"
        else:
            return "Low"

    def fuse(
        self,
        ml_score: float,
        rule_signal: str,
        data_quality: str,
        expected_loss: float,
        priority_score: float,
    ) -> FusedDecision:
        """Resolve final operational state according to the Section 4.6 decision matrix."""
        ml_band = self._determine_ml_band(ml_score)
        rule_sig = rule_signal.capitalize()
        dq = data_quality.capitalize()

        disagreement = False
        reason = ""

        # Safe guard: Poor data quality forces REVIEW if there's any ambiguity or elevated risk
        if dq == "Poor":
            operational_state = "REVIEW"
            confidence = 0.50
            reason = "DATA_QUALITY_POOR: Missing or unverified critical transaction context forces human review."
            if ml_band == "High" or rule_sig == "High":
                disagreement = True

        # ML High + Rule High + Good Quality -> HIGH
        elif ml_band == "High" and rule_sig == "High":
            operational_state = "HIGH"
            confidence = 0.95
            reason = "CONVERGENT_HIGH_RISK: Both ML statistical model and deterministic rules corroborate elevated risk."

        # ML Low + Rule Low + Good Quality -> LOW
        elif ml_band == "Low" and rule_sig == "Low":
            operational_state = "LOW"
            confidence = 0.95
            reason = "CONVERGENT_LOW_RISK: ML model and rules agree transaction matches legitimate baseline."

        # Disagreement Case 1: ML High, Rule Low
        elif ml_band == "High" and rule_sig == "Low":
            operational_state = "REVIEW"
            confidence = 0.70
            disagreement = True
            reason = "SIGNAL_DISAGREEMENT_ML_LEAD: ML model detected complex statistical anomaly, but rules did not fire."

        # Disagreement Case 2: ML Low, Rule High
        elif ml_band == "Low" and rule_sig == "High":
            operational_state = "REVIEW"
            confidence = 0.70
            disagreement = True
            reason = "SIGNAL_DISAGREEMENT_RULE_LEAD: Deterministic rule triggered, but ML overall probability remains low."

        # Rule Corroboration: ML Medium + Rule High
        elif ml_band == "Medium" and rule_sig == "High":
            operational_state = "HIGH"
            confidence = 0.85
            reason = "RULE_ELEVATED_RISK: High deterministic rule severity elevates borderline ML risk score to HIGH."

        # ML Medium + Rule Medium / Low
        elif ml_band == "Medium":
            if expected_loss >= self.review_loss_floor:
                operational_state = "REVIEW"
                confidence = 0.75
                reason = f"EXPOSURE_WEIGHTED_REVIEW: Moderate risk with significant financial exposure (₹{expected_loss:,.2f})."
            else:
                operational_state = "LOW"
                confidence = 0.80
                reason = "MODERATE_LOW_EXPOSURE: Moderate risk but potential loss below review cost floor."

        # ML Low + Rule Medium — rules are mildly elevated but ML is confident it's safe
        elif ml_band == "Low" and rule_sig == "Medium":
            if expected_loss >= self.review_loss_floor:
                operational_state = "REVIEW"
                confidence = 0.65
                reason = f"LOW_ML_MODERATE_RULES: ML confident but rules moderately elevated with significant exposure (₹{expected_loss:,.2f})."
            else:
                operational_state = "LOW"
                confidence = 0.85
                reason = "LOW_ML_MODERATE_RULES_SAFE: ML confident, rules mildly elevated, exposure below floor."

        else:
            operational_state = "REVIEW"
            confidence = 0.60
            reason = "DEFAULT_REVIEW_FALLBACK: Ambiguous signal combination routed to analyst inspection."

        # Recommended Action mapping (Section 2.3)
        if operational_state == "LOW":
            recommended_action = "Allow / normal monitoring"
        elif operational_state == "HIGH":
            recommended_action = "Prioritize investigation / prepare dispute defense"
        else:
            recommended_action = "Human investigation"

        return FusedDecision(
            operational_state=operational_state,
            recommended_action=recommended_action,
            confidence=confidence,
            reason_code=reason,
            ml_band=ml_band,
            rule_signal=rule_sig,
            data_quality=dq,
            expected_loss=expected_loss,
            priority_score=priority_score,
            disagreement_detected=disagreement,
        )
