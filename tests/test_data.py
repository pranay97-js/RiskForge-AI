"""Tests for data ingestion, cleaning, and chronological splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning import clean_transactions, validate_schema
from src.data.loader import generate_benchmark_transactions
from src.data.splitting import chronological_split


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    return generate_benchmark_transactions(
        n_transactions=500,
        n_customers=50,
        random_seed=123,
    )


def test_generate_benchmark_schema(sample_raw_data: pd.DataFrame):
    assert len(sample_raw_data) == 500
    is_valid, errors = validate_schema(sample_raw_data)
    assert is_valid, f"Validation failed with: {errors}"
    assert (sample_raw_data["amount"] > 0).all()
    assert sample_raw_data["is_chargeback"].isin([0, 1]).all()


def test_clean_transactions(sample_raw_data: pd.DataFrame):
    # Introduce messy row
    messy_df = sample_raw_data.copy()
    messy_df.loc[0, "payment_method"] = None
    messy_df.loc[1, "amount"] = -50.0  # Invalid negative amount

    cleaned = clean_transactions(messy_df)
    assert len(cleaned) == 499  # Dropped negative amount row
    assert cleaned["payment_method"].iloc[0] == "unknown"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])


def test_chronological_splitting_no_leakage(sample_raw_data: pd.DataFrame):
    cleaned = clean_transactions(sample_raw_data)
    train_df, val_df, test_df = chronological_split(cleaned, 0.70, 0.15, 0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()
