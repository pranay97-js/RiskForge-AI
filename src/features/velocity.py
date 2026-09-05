"""Velocity feature engineering: tracking short-term transaction frequency without lookahead leakage."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def compute_velocity_features(
    df: pd.DataFrame,
    initial_state: Optional[Dict[str, List[int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Compute rolling customer transaction velocity in the past 1h, 6h, and 24h.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted dataframe with 'customer_id' and 'timestamp'.
    initial_state : dict, optional
        Customer history state from a previous fit/transform call.
        Maps customer_id -> list of prior transaction timestamps (epoch sec).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[int]]]
        Feature dataframe and final customer history state for downstream reuse.
    """
    timestamps = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9  # unix epoch in seconds
    customer_ids = df["customer_id"].values

    n = len(df)
    v_1h = np.zeros(n, dtype=np.int32)
    v_6h = np.zeros(n, dtype=np.int32)
    v_24h = np.zeros(n, dtype=np.int32)

    # Deep copy initial state to avoid mutating the caller's dict
    customer_history: Dict[str, List[int]] = copy.deepcopy(initial_state) if initial_state else {}

    for i in range(n):
        cid = customer_ids[i]
        curr_t = timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i]

        if cid not in customer_history:
            customer_history[cid] = [curr_t]
            continue

        history = customer_history[cid]

        cutoff_24h = curr_t - 86400
        cutoff_6h = curr_t - 21600
        cutoff_1h = curr_t - 3600

        # Count prior transactions falling into respective windows
        count_24h = 0
        count_6h = 0
        count_1h = 0

        for past_t in reversed(history):
            if past_t < cutoff_24h:
                break
            count_24h += 1
            if past_t >= cutoff_6h:
                count_6h += 1
            if past_t >= cutoff_1h:
                count_1h += 1

        v_1h[i] = count_1h
        v_6h[i] = count_6h
        v_24h[i] = count_24h

        # Prune old timestamps beyond 24h window, then append current
        pruned = [t for t in history if t >= cutoff_24h]
        pruned.append(curr_t)
        customer_history[cid] = pruned

    res = pd.DataFrame(
        {
            "velocity_1h": v_1h,
            "velocity_6h": v_6h,
            "velocity_24h": v_24h,
        },
        index=df.index,
    )
    return res, customer_history
