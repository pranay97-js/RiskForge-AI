"""End-to-end feature pipeline: transforms raw transactions into model-ready tabular matrices."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.features.transaction import extract_transaction_features
from src.features.velocity import compute_velocity_features
from src.features.behavioral import compute_behavioral_profiles
from src.features.deviation import compute_deviation_features

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES: List[str] = [
    "payment_method",
    "card_network",
    "merchant_category",
    "device_type",
    "ip_country",
]

NUMERICAL_FEATURES: List[str] = [
    "amount",
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "is_3ds_authenticated",
    "velocity_1h",
    "velocity_6h",
    "velocity_24h",
    "customer_prior_txn_count",
    "customer_historical_mean",
    "customer_historical_std",
    "is_first_transaction",
    "amount_ratio",
    "amount_zscore",
    "velocity_ratio",
]


class RiskFeaturePipeline:
    """Pipelines all transaction, velocity, behavioral, and deviation feature transforms.

    Stores accumulated customer state from fit() so that transform() on
    subsequent splits (val/test) maintains correct behavioral baselines
    and velocity counts instead of resetting to zero.
    """

    def __init__(self):
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        # Customer state accumulated during fit (for cross-split continuity)
        self._velocity_state: Optional[Dict] = None
        self._behavioral_state: Optional[Dict] = None

    def _extract_raw_features(
        self,
        df: pd.DataFrame,
        velocity_state: Optional[Dict] = None,
        behavioral_state: Optional[Dict] = None,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Extract combined dataframe of all sub-feature extractors.

        Returns the feature dataframe plus the final velocity and behavioral
        state dictionaries for reuse in subsequent calls.
        """
        txn_df = extract_transaction_features(df)
        vel_df, vel_state = compute_velocity_features(df, initial_state=velocity_state)
        beh_df, beh_state = compute_behavioral_profiles(df, initial_state=behavioral_state)
        dev_df = compute_deviation_features(df, beh_df, vel_df)

        combined = pd.concat([txn_df, vel_df, beh_df, dev_df], axis=1)
        return combined, vel_state, beh_state

    def fit(self, df: pd.DataFrame) -> RiskFeaturePipeline:
        """Fit categorical encoders, establish model feature columns, and store customer state."""
        combined, vel_state, beh_state = self._extract_raw_features(df)

        # Store accumulated state for use in transform()
        self._velocity_state = vel_state
        self._behavioral_state = beh_state

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.encoder.fit(combined[CATEGORICAL_FEATURES])

        cat_encoded_names = list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = NUMERICAL_FEATURES + cat_encoded_names
        self.is_fitted = True
        logger.info("Fitted Feature Pipeline with %d total features.", len(self.feature_names))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform transactions dataframe into final feature matrix.

        Uses stored customer state from fit() to ensure behavioral baselines
        and velocity counts are continuous across train/val/test splits.
        """
        if not self.is_fitted or self.encoder is None:
            raise RuntimeError("Pipeline must be fitted before calling transform!")

        combined, _, _ = self._extract_raw_features(
            df,
            velocity_state=self._velocity_state,
            behavioral_state=self._behavioral_state,
        )

        num_part = combined[NUMERICAL_FEATURES].astype(np.float32)
        cat_encoded = self.encoder.transform(combined[CATEGORICAL_FEATURES])
        cat_encoded_names = list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        cat_part = pd.DataFrame(cat_encoded, columns=cat_encoded_names, index=df.index, dtype=np.float32)

        final_df = pd.concat([num_part, cat_part], axis=1)
        # Ensure exact column ordering
        return final_df[self.feature_names]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform on training data."""
        return self.fit(df).transform(df)
