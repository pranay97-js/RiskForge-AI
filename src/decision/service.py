"""Unified Risk Decision Service: orchestrates ML, Rules, Cost, SHAP, and Decision Fusion."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import pandas as pd

from src.decision.cost import CostEngine
from src.decision.fusion import DecisionFusionEngine, FusedDecision
from src.explainability.shap import RiskExplainer
from src.features.pipeline import RiskFeaturePipeline
from src.ml.predict import RiskPredictor
from src.rules.engine import RuleEngine, RuleEngineOutput

logger = logging.getLogger(__name__)


class RiskDecisionService:
    """The central runtime service for evaluating transaction risk end-to-end."""

    def __init__(
        self,
        predictor: Optional[RiskPredictor] = None,
        rule_engine: Optional[RuleEngine] = None,
        cost_engine: Optional[CostEngine] = None,
        fusion_engine: Optional[DecisionFusionEngine] = None,
        explainer: Optional[RiskExplainer] = None,
    ):
        self.predictor = predictor or RiskPredictor()
        self.rule_engine = rule_engine or RuleEngine()
        self.cost_engine = cost_engine or CostEngine()
        self.fusion_engine = fusion_engine or DecisionFusionEngine()
        try:
            self.explainer = explainer or RiskExplainer()
        except Exception as e:
            logger.warning("Could not initialize SHAP explainer: %s", e)
            self.explainer = None

    def evaluate_transaction(
        self,
        transaction: Dict[str, Any],
        compute_shap: bool = True,
    ) -> Dict[str, Any]:
        """Execute end-to-end evaluation pipeline for a single transaction.

        Flow:
        1. Preprocess & ML Prediction -> Chargeback Probability
        2. Rule Engine -> Severity Score & Data Quality Score
        3. Cost Engine -> Expected Financial Loss & Priority
        4. Decision Fusion -> Final Operational State (LOW / REVIEW / HIGH)
        5. Explainability (SHAP) -> Top feature drivers (if compute_shap=True)
        6. Return unified audit-friendly dictionary
        """
        # Ensure minimal fields exist
        amount = float(transaction.get("amount", 0.0))
        txn_id = str(transaction.get("transaction_id", "TXN_UNKNOWN"))

        txn_copy = dict(transaction)
        if "timestamp" not in txn_copy or not txn_copy["timestamp"]:
            from datetime import datetime
            txn_copy["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. ML Scoring
        df = pd.DataFrame([txn_copy])
        ml_prob = float(self.predictor.predict_proba(df)[0])


        # 2. Rule Evaluation
        # Pass enriched feature dict to rules, preserving explicit transaction inputs
        feature_df = self.predictor.pipeline.transform(df)
        combined_context = {**feature_df.iloc[0].to_dict(), **transaction}
        rule_output: RuleEngineOutput = self.rule_engine.evaluate(combined_context)

        # 3. Cost Evaluation
        # Tentative review: if ML >= 0.35 or rules >= 0.20
        is_tentative_review = (ml_prob >= 0.35) or (rule_output.rule_severity_score >= 0.20)
        cost_eval = self.cost_engine.evaluate_transaction_cost(
            amount=amount,
            p_chargeback=ml_prob,
            is_reviewed=is_tentative_review,
        )

        # 4. Decision Fusion
        fused: FusedDecision = self.fusion_engine.fuse(
            ml_score=ml_prob,
            rule_signal=rule_output.rule_signal,
            data_quality=rule_output.data_quality,
            expected_loss=cost_eval["expected_loss"],
            priority_score=cost_eval["priority_score"],
        )

        # 5. SHAP Explanations (only when requested, e.g. single-transaction investigation)
        shap_factors = {}
        if compute_shap and self.explainer is not None:
            try:
                shap_factors = self.explainer.explain_transaction(feature_df, top_k=3)
            except Exception as ex:
                logger.warning("SHAP explanation failed for %s: %s", txn_id, ex)

        # 6. Assemble Audit Record
        return {
            "transaction_id": txn_id,
            "operational_decision": fused.to_dict(),
            "ml_risk": {
                "probability": round(ml_prob, 4),
                "band": fused.ml_band,
            },
            "rule_verification": rule_output.to_dict(),
            "financial_exposure": cost_eval,
            "explainability": shap_factors,
        }

