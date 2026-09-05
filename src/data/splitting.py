"""Chronological train/validation/test splitting for RiskForge AI.

Guarantees strict temporal separation to prevent lookahead leakage,
and creates the frozen final test set as required by the production benchmark specification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into Train, Validation, and Test partitions chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataframe sorted by timestamp.
    train_ratio : float
        Proportion for training (default: 0.70).
    val_ratio : float
        Proportion for validation (default: 0.15).
    test_ratio : float
        Proportion for frozen test set (default: 0.15).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Ensure chronological order
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    # Verification of strict temporal bounds
    assert train_df["timestamp"].max() <= val_df["timestamp"].min(), (
        "Temporal leakage: train max timestamp exceeds validation min timestamp!"
    )
    assert val_df["timestamp"].max() <= test_df["timestamp"].min(), (
        "Temporal leakage: validation max timestamp exceeds test min timestamp!"
    )

    logger.info(
        "Chronological Split Complete:\n"
        "  Train: %d rows (%s to %s) | Positives: %d (%.2f%%)\n"
        "  Val:   %d rows (%s to %s) | Positives: %d (%.2f%%)\n"
        "  Test:  %d rows (%s to %s) | Positives: %d (%.2f%%)",
        len(train_df),
        train_df["timestamp"].min(),
        train_df["timestamp"].max(),
        train_df["is_chargeback"].sum(),
        train_df["is_chargeback"].mean() * 100,
        len(val_df),
        val_df["timestamp"].min(),
        val_df["timestamp"].max(),
        val_df["is_chargeback"].sum(),
        val_df["is_chargeback"].mean() * 100,
        len(test_df),
        test_df["timestamp"].min(),
        test_df["timestamp"].max(),
        test_df["is_chargeback"].sum(),
        test_df["is_chargeback"].mean() * 100,
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
) -> Dict[str, str]:
    """Save the train, val, and frozen test splits to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.csv"
    val_file = out_path / "val.csv"
    test_frozen_file = out_path / "test_frozen.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_frozen_file, index=False)

    logger.info("Saved train, val, and frozen test splits to %s", out_path)
    return {
        "train": str(train_file),
        "val": str(val_file),
        "test_frozen": str(test_frozen_file),
    }
