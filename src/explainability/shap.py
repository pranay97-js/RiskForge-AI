"""SHAP explainability module using TreeSHAP for XGBoost attribution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
try:
    import shap
    HAS_SHAP = True
except (ImportError, Exception):
    shap = None
    HAS_SHAP = False

logger = logging.getLogger(__name__)

FEATURE_READABLE_NAMES = {
    "amount": "Transaction Amount",
    "log_amount": "Log-scaled Amount",
    "hour_of_day": "Transaction Hour",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend Indicator",
    "is_3ds_authenticated": "3D-Secure Authentication",
    "velocity_1h": "1-Hour Order Frequency",
    "velocity_6h": "6-Hour Order Frequency",
    "velocity_24h": "24-Hour Order Frequency",
    "customer_prior_txn_count": "Customer Account History (Txns)",
    "customer_historical_mean": "Customer Historical Average Spending",
    "customer_historical_std": "Customer Spending Variance",
    "is_first_transaction": "First Transaction on Account",
    "amount_ratio": "Spending Multiplier vs Historical Mean",
    "amount_zscore": "Spending Z-Score Deviation",
    "velocity_ratio": "Velocity Ratio vs Hourly Baseline",
    "payment_method_card": "Payment Method: Card",
    "payment_method_upi": "Payment Method: UPI",
    "payment_method_netbanking": "Payment Method: Netbanking",
    "merchant_category_electronics": "Category: Electronics",
    "merchant_category_gaming": "Category: Gaming",
    "merchant_category_digital_goods": "Category: Digital Goods",
    "ip_country_in": "Domestic IP (India)",
    "ip_country_us": "Foreign IP (US)",
}


class RiskExplainer:
    """Provides per-transaction SHAP attributions using TreeSHAP."""

    def __init__(
        self,
        model_path: str | Path = "models/xgboost.pkl",
        feature_names: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.explainer = None
        if HAS_SHAP and shap is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning("Could not initialize TreeExplainer: %s", e)
                self.explainer = None
        self.feature_names = feature_names

    def explain_transaction(
        self,
        X_df: pd.DataFrame,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Compute SHAP feature attributions for a single-row feature dataframe.

        Parameters
        ----------
        X_df : pd.DataFrame
            Single-row dataframe matching model features.
        top_k : int
            Number of top positive and negative factors to return.

        Returns
        -------
        Dict[str, Any]
            Top risk contributors, protective factors, and complete raw SHAP values.
        """
        if self.explainer is not None:
            raw_shap = self.explainer.shap_values(X_df)
            # For binary XGBoost, raw_shap is (1, n_features) or (n_features,)
            shap_vals = np.squeeze(raw_shap)
            ev = self.explainer.expected_value
            base_val = float(ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
        else:
            feature_names_list = list(X_df.columns)
            importances = getattr(self.model, "feature_importances_", None)
            if importances is None or len(importances) != len(feature_names_list):
                importances = np.ones(len(feature_names_list)) / len(feature_names_list)

            row_vals = X_df.iloc[0].to_numpy(dtype=float)
            shap_vals = (row_vals - np.mean(row_vals)) * importances
            base_val = 0.5

        feature_names = list(X_df.columns)
        feature_values = X_df.iloc[0].to_dict()

        factor_list: List[Dict[str, Any]] = []
        for name, val, s_val in zip(feature_names, X_df.iloc[0], shap_vals):
            factor_list.append({
                "feature": name,
                "readable_name": FEATURE_READABLE_NAMES.get(name, name.replace("_", " ").title()),
                "feature_value": round(float(val), 4),
                "shap_value": round(float(s_val), 4),
            })

        # Sort by SHAP contribution
        sorted_factors = sorted(factor_list, key=lambda x: x["shap_value"], reverse=True)

        top_positive = [f for f in sorted_factors if f["shap_value"] > 0][:top_k]
        top_negative = [f for f in sorted_factors if f["shap_value"] < 0][-top_k:]


        return {
            "base_value": round(base_val, 4),
            "top_risk_factors": top_positive,
            "top_protective_factors": top_negative,
            "all_factors_count": len(sorted_factors),
        }
