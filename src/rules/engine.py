"""Deterministic rule evaluation engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.rules.registry import RuleRegistry
from src.rules.rules import RuleResult, RuleSeverity

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {
    RuleSeverity.LOW: 0.15,
    RuleSeverity.MEDIUM: 0.30,
    RuleSeverity.HIGH: 0.50,
    RuleSeverity.CRITICAL: 0.85,
}


class RuleEngineOutput:
    """Consolidated report from rule engine evaluation."""

    def __init__(
        self,
        all_results: List[RuleResult],
        triggered_rules: List[RuleResult],
        rule_severity_score: float,
        rule_signal: str,
        data_quality_score: float,
        data_quality: str,
    ):
        self.all_results = all_results
        self.triggered_rules = triggered_rules
        self.rule_severity_score = rule_severity_score
        self.rule_signal = rule_signal
        self.data_quality_score = data_quality_score
        self.data_quality = data_quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered_rule_ids": [r.rule_id for r in self.triggered_rules],
            "triggered_rules": [r.to_dict() for r in self.triggered_rules],
            "rule_severity_score": round(self.rule_severity_score, 4),
            "rule_signal": self.rule_signal,
            "data_quality_score": round(self.data_quality_score, 4),
            "data_quality": self.data_quality,
        }


class RuleEngine:
    """Executes deterministic rules against transaction payloads."""

    def __init__(self, registry: Optional[RuleRegistry] = None):
        self.registry = registry or RuleRegistry.get_default_registry()

    def evaluate(
        self,
        transaction: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RuleEngineOutput:
        """Evaluate all active rules on the transaction."""
        rules = self.registry.list_rules(only_enabled=True)
        results: List[RuleResult] = []
        triggered: List[RuleResult] = []

        total_severity_points = 0.0
        data_insufficient_triggered = False

        for rule in rules:
            res = rule.evaluate(transaction, context)
            results.append(res)
            if res.triggered:
                triggered.append(res)
                total_severity_points += SEVERITY_WEIGHTS.get(res.severity, 0.2)
                if res.rule_id == "DATA_INSUFFICIENT":
                    data_insufficient_triggered = True

        # Normalized cumulative rule severity score in [0.0, 1.0]
        rule_severity_score = min(1.0, total_severity_points)

        # Map to discrete rule signal
        if rule_severity_score >= 0.50:
            rule_signal = "High"
        elif rule_severity_score >= 0.20:
            rule_signal = "Medium"
        else:
            rule_signal = "Low"

        # Data quality score
        if data_insufficient_triggered:
            data_quality_score = 0.40
            data_quality = "Poor"
        else:
            data_quality_score = 1.00
            data_quality = "Good"

        return RuleEngineOutput(
            all_results=results,
            triggered_rules=triggered,
            rule_severity_score=rule_severity_score,
            rule_signal=rule_signal,
            data_quality_score=data_quality_score,
            data_quality=data_quality,
        )
