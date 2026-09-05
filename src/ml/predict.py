"""Inference engine for RiskForge AI models."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd


from src.features.pipeline import RiskFeaturePipeline

logger = logging.getLogger(__name__)


class RiskPredictor:
    """Production inference interface for scoring chargeback risk on raw transactions."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        pipeline_path: str | Path = "models/feature_pipeline.pkl",
    ):
        self.pipeline_path = Path(pipeline_path)
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {self.pipeline_path}")

        logger.info("Loading feature pipeline from %s...", self.pipeline_path)
        self.pipeline: RiskFeaturePipeline = joblib.load(self.pipeline_path)

        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            xgboost_path = Path("models/xgboost.pkl")
            baseline_path = Path("models/baseline.pkl")
            has_xgboost = False
            try:
                import xgboost  # noqa: F401
                has_xgboost = True
            except (ImportError, Exception):
                has_xgboost = False

            if has_xgboost and xgboost_path.exists():
                self.model_path = xgboost_path
            elif baseline_path.exists():
                logger.info("Using calibrated baseline pipeline (serverless mode).")
                self.model_path = baseline_path
            else:
                self.model_path = xgboost_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        logger.info("Loading model from %s...", self.model_path)
        self.model = joblib.load(self.model_path)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Score a batch of transactions and return probability of chargeback."""
        X = self.pipeline.transform(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary prediction based on operational threshold."""
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)

    def predict_single(self, transaction: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        """Score a single transaction payload."""
        df = pd.DataFrame([transaction])
        prob = float(self.predict_proba(df)[0])

        if prob >= 0.65:
            level = "HIGH"
        elif prob >= 0.25:
            level = "REVIEW"
        else:
            level = "LOW"

        return {
            "transaction_id": transaction.get("transaction_id", "UNKNOWN"),
            "risk_score": round(prob, 4),
            "risk_level": level,
            "threshold_used": threshold,
            "is_flagged": prob >= threshold,
        }
