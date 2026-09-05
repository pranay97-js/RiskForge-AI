"""Behavioral deviation features: measuring individual anomaly against historical baseline.

Corresponds directly to Section 4.3 'Behavioral Deviation Score' of the blueprint:
- Amount ratio = current_amount / (customer_historical_average + eps)
- Amount z-score = (current_amount - customer_mean) / (customer_std + eps)
- Velocity ratio = current_velocity / (historical_velocity + eps)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_deviation_features(
    df: pd.DataFrame,
    behavioral_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    eps: float = 1e-5,
) -> pd.DataFrame:
    """Compute mathematical deviation metrics comparing current transaction to historical baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe containing 'amount'.
    behavioral_df : pd.DataFrame
        Output of `compute_behavioral_profiles` with historical mean and std.
    velocity_df : pd.DataFrame
        Output of `compute_velocity_features` with velocity counters.
    eps : float
        Numerical stability epsilon to prevent division by zero.

    Returns
    -------
    pd.DataFrame
        Dataframe containing 'amount_ratio', 'amount_zscore', and 'velocity_ratio'.
    """
    amount = df["amount"].values.astype(np.float64)
    hist_mean = behavioral_df["customer_historical_mean"].values.astype(np.float64)
    hist_std = behavioral_df["customer_historical_std"].values.astype(np.float64)
    v_1h = velocity_df["velocity_1h"].values.astype(np.float64)
    v_24h = velocity_df["velocity_24h"].values.astype(np.float64)

    # 1. Amount ratio
    amount_ratio = amount / (hist_mean + eps)

    # 2. Amount z-score
    amount_zscore = (amount - hist_mean) / (hist_std + eps)

    # 3. Velocity ratio (comparing 1h burst against 24h hourly average)
    hourly_baseline_velocity = (v_24h / 24.0)
    velocity_ratio = v_1h / (hourly_baseline_velocity + 0.1)

    res = pd.DataFrame(
        {
            "amount_ratio": np.clip(amount_ratio, 0.0, 100.0),
            "amount_zscore": np.clip(amount_zscore, -5.0, 50.0),
            "velocity_ratio": np.clip(velocity_ratio, 0.0, 50.0),
        },
        index=df.index,
    )
    return res
