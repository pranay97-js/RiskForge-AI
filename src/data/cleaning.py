"""Data cleaning, integrity checks, and validation routines for RiskForge AI."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = [
    "transaction_id",
    "timestamp",
    "customer_id",
    "amount",
    "payment_method",
    "card_network",
    "merchant_category",
    "device_type",
    "ip_country",
    "is_3ds_authenticated",
    "is_chargeback",
]


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that incoming transaction DataFrame conforms to expected schema.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors: List[str] = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if "amount" in df.columns:
        if (df["amount"] <= 0).any():
            errors.append("Dataset contains non-positive amounts.")

    return len(errors) == 0, errors


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize transactions table.

    Steps:
    1. Copy dataframe to avoid mutating source.
    2. Convert timestamp column to datetime.
    3. Sort strictly by timestamp ascending.
    4. Fill missing categorical values with 'unknown'.
    5. Ensure binary flags are cleanly typed integers/booleans.
    """
    clean_df = df.copy()

    # Convert timestamps
    if not pd.api.types.is_datetime64_any_dtype(clean_df["timestamp"]):
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

    # Ensure chronological order
    clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)

    # Impute categorical missing values
    cat_columns = [
        "payment_method",
        "card_network",
        "merchant_category",
        "device_type",
        "ip_country",
    ]
    for col in cat_columns:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna("unknown").astype(str).str.lower()

    # Ensure amount is float and strictly positive (filter NaN and Inf)
    clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce")
    clean_df = clean_df[np.isfinite(clean_df["amount"]) & (clean_df["amount"] > 0)].copy()

    # Ensure boolean/integer flags
    if "is_3ds_authenticated" in clean_df.columns:
        clean_df["is_3ds_authenticated"] = clean_df["is_3ds_authenticated"].astype(bool)

    if "is_chargeback" in clean_df.columns:
        clean_df["is_chargeback"] = clean_df["is_chargeback"].astype(int)

    logger.info("Cleaned %d transactions successfully.", len(clean_df))
    return clean_df
