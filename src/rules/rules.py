"""Deterministic rule definitions for transaction risk corroboration.

Corresponds to Section 4.5 of the blueprint:
Rules are independent, deterministic, auditable, and versioned checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class RuleSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleResult:
    """Outcome of a single deterministic rule evaluation."""

    def __init__(
        self,
        rule_id: str,
        category: str,
        triggered: bool,
        severity: RuleSeverity,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.rule_id = rule_id
        self.category = category
        self.triggered = triggered
        self.severity = severity
        self.description = description
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "triggered": self.triggered,
            "severity": self.severity.value,
            "description": self.description,
            "details": self.details,
        }


class BaseRule(ABC):
    """Abstract base class for all deterministic rules."""

    def __init__(
        self,
        rule_id: str,
        category: str,
        severity: RuleSeverity,
        version: str = "1.0.0",
        description: str = "",
    ):
        self.rule_id = rule_id
        self.category = category
        self.severity = severity
        self.version = version
        self.description = description

    @abstractmethod
    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        """Evaluate rule logic on the transaction and its calculated features."""
        pass


class VelocityAnomalyRule(BaseRule):
    """Rule 1: Triggers if recent transaction velocity spikes anomalously."""

    def __init__(
        self,
        velocity_1h_threshold: int = 3,
        velocity_ratio_threshold: float = 3.0,
        version: str = "1.0.0",
    ):
        super().__init__(
            rule_id="VELOCITY_ANOMALY",
            category="Behavioral",
            severity=RuleSeverity.HIGH,
            version=version,
            description="Transaction frequency spikes significantly above customer baseline.",
        )
        self.velocity_1h_threshold = velocity_1h_threshold
        self.velocity_ratio_threshold = velocity_ratio_threshold

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        v1h = int(transaction.get("velocity_1h", 0))
        v_ratio = float(transaction.get("velocity_ratio", 0.0))

        triggered = (v1h >= self.velocity_1h_threshold) or (v_ratio >= self.velocity_ratio_threshold)
        return RuleResult(
            rule_id=self.rule_id,
            category=self.category,
            triggered=triggered,
            severity=self.severity,
            description=self.description,
            details={"velocity_1h": v1h, "velocity_ratio": v_ratio},
        )


class SpendingDeviationRule(BaseRule):
    """Rule 2: Triggers if transaction amount materially exceeds historical baseline."""

    def __init__(
        self,
        amount_ratio_threshold: float = 3.5,
        amount_zscore_threshold: float = 3.5,
        min_amount_floor: float = 5000.0,
        version: str = "1.0.0",
    ):
        super().__init__(
            rule_id="SPENDING_DEVIATION",
            category="Behavioral",
            severity=RuleSeverity.HIGH,
            version=version,
            description="Transaction amount materially exceeds customer historical baseline.",
        )
        self.amount_ratio_threshold = amount_ratio_threshold
        self.amount_zscore_threshold = amount_zscore_threshold
        self.min_amount_floor = min_amount_floor

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        amount = float(transaction.get("amount", 0.0))
        ratio = float(transaction.get("amount_ratio", 1.0))
        zscore = float(transaction.get("amount_zscore", 0.0))

        # Only trigger deviation if amount is non-trivial
        triggered = (amount >= self.min_amount_floor) and (
            (ratio >= self.amount_ratio_threshold) or (zscore >= self.amount_zscore_threshold)
        )
        return RuleResult(
            rule_id=self.rule_id,
            category=self.category,
            triggered=triggered,
            severity=self.severity,
            description=self.description,
            details={"amount": amount, "amount_ratio": ratio, "amount_zscore": zscore},
        )


class DisputeHistoryRule(BaseRule):
    """Rule 3: Triggers if customer entity has prior dispute or chargeback history."""

    def __init__(self, version: str = "1.0.0"):
        super().__init__(
            rule_id="DISPUTE_HISTORY",
            category="History",
            severity=RuleSeverity.CRITICAL,
            version=version,
            description="Prior dispute or chargeback record is associated with this entity.",
        )

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        prior_disputes = int(transaction.get("prior_dispute_count", 0))
        has_chargeback_flag = bool(transaction.get("has_prior_chargeback", False))

        triggered = (prior_disputes > 0) or has_chargeback_flag
        return RuleResult(
            rule_id=self.rule_id,
            category=self.category,
            triggered=triggered,
            severity=self.severity,
            description=self.description,
            details={"prior_disputes": prior_disputes, "has_prior_chargeback": has_chargeback_flag},
        )


class NewAccountHighValueRule(BaseRule):
    """Rule 4: Triggers when a new account attempts an unusually high-value first purchase."""

    def __init__(self, high_value_threshold: float = 20000.0, version: str = "1.0.0"):
        super().__init__(
            rule_id="NEW_ACCOUNT_HIGH_VALUE",
            category="History",
            severity=RuleSeverity.MEDIUM,
            version=version,
            description="New account combined with unusually high-value transaction.",
        )
        self.high_value_threshold = high_value_threshold

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        is_first = bool(transaction.get("is_first_transaction", False))
        amount = float(transaction.get("amount", 0.0))

        triggered = is_first and (amount >= self.high_value_threshold)
        return RuleResult(
            rule_id=self.rule_id,
            category=self.category,
            triggered=triggered,
            severity=self.severity,
            description=self.description,
            details={"is_first_transaction": is_first, "amount": amount},
        )


class PaymentChangeRule(BaseRule):
    """Rule 5: Triggers on unexpected changes in payment context (e.g. non-3DS + foreign IP)."""

    def __init__(self, version: str = "1.0.0"):
        super().__init__(
            rule_id="PAYMENT_CHANGE",
            category="Consistency",
            severity=RuleSeverity.MEDIUM,
            version=version,
            description="Unexpected payment context change without 3D-Secure authentication.",
        )

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        is_3ds = bool(transaction.get("is_3ds_authenticated", True))
        ip_country = str(transaction.get("ip_country", "IN")).upper()
        amount = float(transaction.get("amount", 0.0))

        # Trigger if non-3DS authenticated and cross-border IP or elevated value
        triggered = (not is_3ds) and (ip_country not in ["IN", "UNKNOWN"] or amount >= 15000.0)
        return RuleResult(
            rule_id=self.rule_id,
            category=self.category,
            triggered=triggered,
            severity=self.severity,
            description=self.description,
            details={"is_3ds_authenticated": is_3ds, "ip_country": ip_country, "amount": amount},
        )


class DataInsufficientRule(BaseRule):
    """Rule 6: Triggers when critical transaction context is missing or untrusted."""

    def __init__(self, version: str = "1.0.0"):
        super().__init__(
            rule_id="DATA_INSUFFICIENT",
            category="Data Quality",
            severity=RuleSeverity.LOW,
            version=version,
            description="Required transaction context is missing or unreliable.",
        )

    def evaluate(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> RuleResult:
        required_fields = ["amount", "customer_id", "payment_method"]
        missing = [f for f in required_fields if transaction.get(f) is None or str(transaction.get(f)).strip() == ""]
        
        has_amount = float(transaction.get("amount", 0.0)) > 0.0
        if not has_amount and "amount" not in missing:
            missing.append("valid_amount_gt_zero")

        triggered = len(missing) > 0
        return RuleResult(
            rule_id=self.rule_id,
            category=self.category,
            triggered=triggered,
            severity=self.severity,
            description=self.description,
            details={"missing_fields": missing},
        )
