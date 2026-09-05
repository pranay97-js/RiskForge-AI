"""Unit tests for the financial cost engine and threshold sweep optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.cost import CostEngine
from src.decision.threshold import sweep_cost_thresholds


def test_cost_engine_expected_loss():
    engine = CostEngine(investigation_cost=50.0, chargeback_penalty_fee=3000.0)
    # Amount ₹10,000 with 50% probability
    # Potential loss = 10,000 + 3,000 = 13,000
    # Expected loss = 0.5 * 13,000 = 6,500
    exp_loss = engine.calculate_expected_loss(10000.0, 0.5)
    assert exp_loss == 6500.0


def test_cost_engine_transaction_evaluation():
    engine = CostEngine(investigation_cost=50.0, chargeback_penalty_fee=3000.0, preventable_loss_rate=0.90)

    # Reviewed high risk
    res_reviewed = engine.evaluate_transaction_cost(amount=10000.0, p_chargeback=0.8, is_reviewed=True)
    assert res_reviewed["is_reviewed"] is True
    assert res_reviewed["expected_loss_prevented"] > 0
    assert res_reviewed["net_operational_gain"] > 0
    assert res_reviewed["priority_score"] > 0

    # Unreviewed low risk
    res_unreviewed = engine.evaluate_transaction_cost(amount=1000.0, p_chargeback=0.01, is_reviewed=False)
    assert res_unreviewed["expected_loss_prevented"] == 0.0
    assert res_unreviewed["investigation_cost"] == 50.0


def test_sweep_cost_thresholds():
    engine = CostEngine(investigation_cost=50.0, chargeback_penalty_fee=3000.0)

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.85, 0.9, 0.95, 0.1, 0.2, 0.8])
    amounts = np.array([1000, 2000, 1500, 800, 12000, 25000, 18000, 3000, 1200, 15000])

    sweep = sweep_cost_thresholds(y_true, y_prob, amounts, engine)
    assert "optimal_threshold" in sweep
    assert 0.05 <= sweep["optimal_threshold"] <= 0.95
    assert len(sweep["cost_curve"]) > 0
