"""Pydantic structured schemas for Grounded AI input/output contracts.

Corresponds to Section 5.2 and 5.3 of the blueprint:
Enforces strict typing and zero hallucination boundaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class EvidenceChecklist(BaseModel):
    """Structured checklist of merchant evidence for chargeback dispute defence."""
    payment_proof: bool = Field(default=False, description="Verified payment receipt or gateway payment ID")
    three_ds_auth: bool = Field(default=False, description="3D-Secure OTP or biometric authentication log")
    delivery_proof: bool = Field(default=False, description="Courier tracking or signed proof of delivery")
    customer_communication: bool = Field(default=False, description="Order confirmation email or chat logs")

    def completeness_percentage(self) -> float:
        total = 4
        available = sum([
            self.payment_proof,
            self.three_ds_auth,
            self.delivery_proof,
            self.customer_communication,
        ])
        return round((available / total) * 100.0, 1)

    def is_dispute_defensible(self) -> bool:
        # Defensible if payment, 3DS, and delivery proof are all present
        return bool(self.payment_proof and self.three_ds_auth and self.delivery_proof)


class GroundedCaseInput(BaseModel):
    """Input payload passed to the grounded AI explanation layer."""
    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(description="LOW, REVIEW, or HIGH")
    rules_triggered: List[str] = Field(default_factory=list)
    top_model_factors: List[str] = Field(default_factory=list)
    evidence: EvidenceChecklist = Field(default_factory=EvidenceChecklist)
    expected_loss: float = Field(ge=0.0)
    decision: str = Field(description="Operational state from fusion engine: LOW, REVIEW, or HIGH")
    customer_id: Optional[str] = None
    amount: Optional[float] = None

    @field_validator("transaction_id", "customer_id", mode="before")
    @classmethod
    def sanitize_id_fields(cls, v: Optional[str]) -> Optional[str]:
        """Strip any characters that could be used for prompt injection.

        Only alphanumeric, underscore, hyphen, dot are allowed.
        Length is capped at 64 characters to prevent payload stuffing.
        """
        if v is None:
            return v
        # Allow only alphanumeric, underscore, hyphen, dot
        sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "", str(v))
        # Cap length to prevent payload injection via long strings
        sanitized = sanitized[:64]
        return sanitized or "UNKNOWN"


class GroundedCaseOutput(BaseModel):
    """Strict structured output emitted by the grounded AI layer."""
    transaction_id: str
    case_summary: str = Field(description="Concise 1-2 sentence executive summary of the case")
    flag_reasons: List[str] = Field(description="List of specific mathematical factors and rules that triggered risk")
    evidence_status: Dict[str, str] = Field(description="Map of evidence items and their availability status")
    evidence_completeness_score: float = Field(description="Percentage completeness [0-100%]")
    recommended_action: str = Field(description="Recommended merchant operational next step")
    dispute_response_draft: Optional[str] = Field(
        default=None,
        description="Formal chargeback dispute defense letter, generated only when evidence is sufficient"
    )
    is_defense_ready: bool = Field(description="Whether merchant has sufficient proof to dispute")
    model_version: str = Field(default="RiskForge-Grounded-v1.0")
