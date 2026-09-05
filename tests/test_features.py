"""Tests for feature engineering modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import generate_benchmark_transactions
from src.features.behavioral import compute_behavioral_profiles
from src.features.deviation import compute_deviation_features
from src.features.pipeline import RiskFeaturePipeline
from src.features.transaction import extract_transaction_features
from src.features.velocity import compute_velocity_features


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    return generate_benchmark_transactions(
        n_transactions=300,
        n_customers=30,
        random_seed=42,
    )


def test_extract_transaction_features(transactions_df: pd.DataFrame):
    feat_df = extract_transaction_features(transactions_df)
    assert "hour_sin" in feat_df.columns
    assert "hour_cos" in feat_df.columns
    assert "log_amount" in feat_df.columns
    assert (feat_df["hour_sin"] >= -1.0).all() and (feat_df["hour_sin"] <= 1.0).all()
    assert (feat_df["log_amount"] > 0).all()


def test_velocity_features_monotonic(transactions_df: pd.DataFrame):
    vel_df, _ = compute_velocity_features(transactions_df)
    assert (vel_df["velocity_1h"] >= 0).all()
    assert (vel_df["velocity_6h"] >= vel_df["velocity_1h"]).all()
    assert (vel_df["velocity_24h"] >= vel_df["velocity_6h"]).all()


def test_behavioral_profiles_first_transaction(transactions_df: pd.DataFrame):
    beh_df, _ = compute_behavioral_profiles(transactions_df)
    assert "customer_prior_txn_count" in beh_df.columns
    assert "is_first_transaction" in beh_df.columns

    # First transaction for a customer must have prior count 0
    first_rows = beh_df[beh_df["is_first_transaction"] == 1]
    assert (first_rows["customer_prior_txn_count"] == 0).all()
    assert not beh_df.isnull().any().any()


def test_deviation_features_no_nan_inf(transactions_df: pd.DataFrame):
    vel_df, _ = compute_velocity_features(transactions_df)
    beh_df, _ = compute_behavioral_profiles(transactions_df)
    dev_df = compute_deviation_features(transactions_df, beh_df, vel_df)

    assert not np.isnan(dev_df.values).any()
    assert not np.isinf(dev_df.values).any()
    assert (dev_df["amount_ratio"] >= 0).all()


def test_risk_feature_pipeline_fit_transform(transactions_df: pd.DataFrame):
    pipeline = RiskFeaturePipeline()
    X = pipeline.fit_transform(transactions_df)

    assert len(X) == len(transactions_df)
    assert len(pipeline.feature_names) > 20
    assert not X.isnull().any().any()

    # Transform new data without refitting
    new_data = transactions_df.head(10)
    X_new = pipeline.transform(new_data)
    assert len(X_new) == 10
    assert list(X_new.columns) == pipeline.feature_names
