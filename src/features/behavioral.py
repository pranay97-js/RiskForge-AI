"""Behavioral feature engineering: customer profile baselines computed without lookahead."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_behavioral_profiles(
    df: pd.DataFrame,
    default_mean_amount: float = 2500.0,
    default_std_amount: float = 1500.0,
    initial_state: Optional[Dict[str, Tuple[float, float, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float, int]]]:
    """Compute expanding customer historical baselines strictly using past transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted dataframe with 'customer_id' and 'amount'.
    default_mean_amount : float
        Population fallback mean for cold-start customers.
    default_std_amount : float
        Population fallback standard deviation for cold-start customers.
    initial_state : dict, optional
        Customer state from a previous fit/transform call.
        Maps customer_id -> (sum_x, sum_x2, count).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Tuple[float, float, int]]]
        Feature dataframe and final customer state for downstream reuse.
    """
    n = len(df)
    customer_ids = df["customer_id"].values
    amounts = df["amount"].values.astype(np.float64)

    prior_counts = np.zeros(n, dtype=np.int32)
    hist_means = np.zeros(n, dtype=np.float64)
    hist_stds = np.zeros(n, dtype=np.float64)
    is_first_txn = np.zeros(n, dtype=np.int32)

    # Deep copy to avoid mutating caller state
    state: Dict[str, Tuple[float, float, int]] = copy.deepcopy(initial_state) if initial_state else {}

    for i in range(n):
        cid = customer_ids[i]
        amt = amounts[i]

        if cid not in state:
            prior_counts[i] = 0
            hist_means[i] = default_mean_amount
            hist_stds[i] = default_std_amount
            is_first_txn[i] = 1
            # Initialize with current transaction
            state[cid] = (amt, amt * amt, 1)
        else:
            sum_x, sum_x2, count = state[cid]
            prior_counts[i] = count
            mean_val = sum_x / count
            if count > 1:
                variance = max(0.0, (sum_x2 / count) - (mean_val * mean_val))
                std_val = float(np.sqrt(variance))
                if std_val < 10.0:  # Bound minimum variance
                    std_val = default_std_amount * 0.2
            else:
                std_val = default_std_amount

            hist_means[i] = mean_val
            hist_stds[i] = std_val
            is_first_txn[i] = 0

            # Update running state
            state[cid] = (sum_x + amt, sum_x2 + (amt * amt), count + 1)

    res = pd.DataFrame(
        {
            "customer_prior_txn_count": prior_counts,
            "customer_historical_mean": hist_means,
            "customer_historical_std": hist_stds,
            "is_first_transaction": is_first_txn,
        },
        index=df.index,
    )
    return res, state
