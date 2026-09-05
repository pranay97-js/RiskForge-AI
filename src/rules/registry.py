"""Rule registry for managing and discovering deterministic verification rules."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.rules.rules import (
    BaseRule,
    DataInsufficientRule,
    DisputeHistoryRule,
    NewAccountHighValueRule,
    PaymentChangeRule,
    SpendingDeviationRule,
    VelocityAnomalyRule,
)

logger = logging.getLogger(__name__)


class RuleRegistry:
    """Central registry holding active deterministic rules."""

    def __init__(self):
        self._rules: Dict[str, BaseRule] = {}
        self._enabled: Dict[str, bool] = {}

    def register(self, rule: BaseRule, enabled: bool = True) -> None:
        """Register a new deterministic rule."""
        self._rules[rule.rule_id] = rule
        self._enabled[rule.rule_id] = enabled
        logger.debug("Registered rule %s (enabled=%s)", rule.rule_id, enabled)

    def unregister(self, rule_id: str) -> None:
        """Remove a rule from registry."""
        self._rules.pop(rule_id, None)
        self._enabled.pop(rule_id, None)

    def set_enabled(self, rule_id: str, enabled: bool) -> None:
        """Enable or disable an existing rule."""
        if rule_id in self._rules:
            self._enabled[rule_id] = enabled

    def get_rule(self, rule_id: str) -> Optional[BaseRule]:
        return self._rules.get(rule_id)

    def list_rules(self, only_enabled: bool = True) -> List[BaseRule]:
        """List all registered rules."""
        if only_enabled:
            return [rule for r_id, rule in self._rules.items() if self._enabled.get(r_id, False)]
        return list(self._rules.values())

    @classmethod
    def get_default_registry(cls) -> RuleRegistry:
        """Instantiate registry with all 6 default rules defined in the blueprint."""
        reg = cls()
        reg.register(VelocityAnomalyRule())
        reg.register(SpendingDeviationRule())
        reg.register(DisputeHistoryRule())
        reg.register(NewAccountHighValueRule())
        reg.register(PaymentChangeRule())
        reg.register(DataInsufficientRule())
        return reg
