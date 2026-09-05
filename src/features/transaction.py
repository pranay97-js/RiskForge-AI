"""Transaction-level and contextual feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract context, cyclical temporal, and amount transformation features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'timestamp', 'amount', and categoricals.

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with new transaction features.
    """
    res = pd.DataFrame(index=df.index)

    # 1. Temporal cyclical features
    dt_series = pd.to_datetime(df["timestamp"])
    hour = dt_series.dt.hour
    dayofweek = dt_series.dt.dayofweek

    res["hour_of_day"] = hour
    res["day_of_week"] = dayofweek
    res["is_weekend"] = dayofweek.isin([5, 6]).astype(int)

    # Cyclical hour encoding (smooth 24-hour cycle)
    res["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    res["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # 2. Amount transformations
    res["amount"] = df["amount"].astype(float)
    res["log_amount"] = np.log1p(df["amount"])

    # 3. Authentication & security flags
    if "is_3ds_authenticated" in df.columns:
        res["is_3ds_authenticated"] = df["is_3ds_authenticated"].astype(int)
    else:
        res["is_3ds_authenticated"] = 1

    # 4. Categoricals
    cat_cols = [
        "payment_method",
        "card_network",
        "merchant_category",
        "device_type",
        "ip_country",
    ]
    for col in cat_cols:
        if col in df.columns:
            res[col] = df[col].astype(str)
        else:
            res[col] = "unknown"

    return res
