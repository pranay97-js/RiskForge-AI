"""Financial cost engine and expected-loss modeling.

Implements Section 4.7 of the blueprint:
- Expected loss = P(chargeback) * potential loss
- Operational trade-off between manual investigation cost and dispute losses.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np


class CostEngine:
    """Calculates expected financial loss, review utility, and investigation queue priority."""

    def __init__(
        self,
        investigation_cost: float = 50.0,
        chargeback_penalty_fee: float = 3000.0,
        preventable_loss_rate: float = 0.90,
    ):
        """
        Parameters
        ----------
        investigation_cost : float
            Labor cost to manually review a flagged transaction (default: ₹50).
        chargeback_penalty_fee : float
            Bank/gateway dispute assessment penalty fee (default: ₹3,000).
        preventable_loss_rate : float
            Portion of loss recovered/prevented when a dispute is caught early (default: 90%).
        """
        self.investigation_cost = float(investigation_cost)
        self.chargeback_penalty_fee = float(chargeback_penalty_fee)
        self.preventable_loss_rate = float(preventable_loss_rate)

    def calculate_expected_loss(
        self,
        amount: float,
        p_chargeback: float,
    ) -> float:
        """Expected direct loss if no defensive action is taken."""
        potential_loss = amount + self.chargeback_penalty_fee
        return float(p_chargeback * potential_loss)

    def evaluate_transaction_cost(
        self,
        amount: float,
        p_chargeback: float,
        is_reviewed: bool,
    ) -> Dict[str, Any]:
        """Evaluate expected cost, potential savings, and net operational utility."""
        potential_loss = amount + self.chargeback_penalty_fee
        expected_loss_unreviewed = p_chargeback * potential_loss

        if is_reviewed:
            # Review cost incurred
            investigation_burden = self.investigation_cost
            # Prevented loss given probability of chargeback
            expected_loss_prevented = p_chargeback * potential_loss * self.preventable_loss_rate
            net_operational_gain = expected_loss_prevented - investigation_burden
            residual_loss = p_chargeback * potential_loss * (1.0 - self.preventable_loss_rate)
            total_expected_cost = investigation_burden + residual_loss
        else:
            investigation_burden = 0.0
            expected_loss_prevented = 0.0
            net_operational_gain = 0.0
            total_expected_cost = expected_loss_unreviewed

        # Priority score for queue ordering
        priority_score = expected_loss_unreviewed / (self.investigation_cost + 1.0)

        return {
            "amount": round(amount, 2),
            "p_chargeback": round(p_chargeback, 4),
            "potential_loss": round(potential_loss, 2),
            "expected_loss": round(expected_loss_unreviewed, 2),
            "investigation_cost": self.investigation_cost,
            "is_reviewed": is_reviewed,
            "expected_loss_prevented": round(expected_loss_prevented, 2),
            "net_operational_gain": round(net_operational_gain, 2),
            "total_expected_cost": round(total_expected_cost, 2),
            "priority_score": round(priority_score, 2),
        }

    def evaluate_portfolio_cost(
        self,
        amounts: np.ndarray,
        p_chargebacks: np.ndarray,
        review_flags: np.ndarray,
    ) -> Dict[str, float]:
        """Aggregate total financial loss across an entire batch or portfolio."""
        amounts_arr = np.asarray(amounts, dtype=float)
        probs_arr = np.asarray(p_chargebacks, dtype=float)
        reviews_arr = np.asarray(review_flags, dtype=bool)

        potential_losses = amounts_arr + self.chargeback_penalty_fee
        baseline_losses = probs_arr * potential_losses

        # Costs under policy
        review_costs = reviews_arr.astype(float) * self.investigation_cost
        prevented_losses = reviews_arr.astype(float) * probs_arr * potential_losses * self.preventable_loss_rate
        unprevented_losses = baseline_losses - prevented_losses
        total_policy_costs = review_costs + unprevented_losses

        total_baseline_loss = float(baseline_losses.sum())
        total_policy_cost = float(total_policy_costs.sum())
        net_savings = total_baseline_loss - total_policy_cost

        return {
            "total_portfolio_exposure": round(total_baseline_loss, 2),
            "total_investigation_costs": round(float(review_costs.sum()), 2),
            "total_prevented_losses": round(float(prevented_losses.sum()), 2),
            "total_net_operational_cost": round(total_policy_cost, 2),
            "net_avoidable_savings": round(net_savings, 2),
            "review_rate": round(float(reviews_arr.mean()), 4),
        }
