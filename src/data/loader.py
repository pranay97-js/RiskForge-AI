"""Dataset loader and benchmark generator for RiskForge AI.

Supports loading raw transaction data or generating a high-fidelity benchmark
dataset simulating e-commerce/Razorpay chargeback loss scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_benchmark_transactions(
    n_transactions: int = 40000,
    n_customers: int = 4000,
    random_seed: int = 42,
    target_chargeback_rate: float = 0.025,
) -> pd.DataFrame:
    """Generate a realistic synthetic transaction dataset adhering to Razorpay / IEEE-CIS schema.

    Parameters
    ----------
    n_transactions : int
        Total number of chronological transactions to generate.
    n_customers : int
        Number of unique customer entities to allow behavioral baseline modeling.
    random_seed : int
        Seed for reproducibility.
    target_chargeback_rate : float
        Proportion of transactions that result in chargebacks/disputes (~2.5%).

    Returns
    -------
    pd.DataFrame
        Synthetic transaction dataset sorted chronologically.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Customer Profiles: baseline mean spending & variance
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    # Log-normal customer average amounts (e.g. ₹500 to ₹15,000)
    customer_mean_amounts = np.exp(rng.normal(loc=7.5, scale=0.8, size=n_customers))
    customer_std_amounts = customer_mean_amounts * rng.uniform(0.15, 0.45, size=n_customers)
    customer_profiles = {
        cid: (mean_val, std_val)
        for cid, mean_val, std_val in zip(customer_ids, customer_mean_amounts, customer_std_amounts)
    }

    # 2. Chronological Timestamps (spanning 60 days)
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    # Time delta between transactions in seconds (exponential arrival)
    seconds_deltas = rng.exponential(scale=130.0, size=n_transactions).cumsum()
    timestamps = [start_time + pd.Timedelta(seconds=float(s)) for s in seconds_deltas]

    # Assign customers with Power-Law / Pareto distribution
    customer_weights = 1.0 / (np.arange(1, n_customers + 1) ** 0.6)
    customer_weights /= customer_weights.sum()
    chosen_customers = rng.choice(customer_ids, size=n_transactions, p=customer_weights)

    # Categorical distributions
    payment_methods = ["upi", "card", "netbanking", "wallet"]
    pm_weights = [0.55, 0.30, 0.10, 0.05]

    card_networks = ["visa", "mastercard", "rupay", "amex", "unknown"]
    cn_weights = [0.45, 0.35, 0.15, 0.03, 0.02]

    merchant_categories = ["electronics", "digital_goods", "fashion", "travel", "groceries", "gaming"]
    mc_weights = [0.20, 0.15, 0.25, 0.10, 0.20, 0.10]

    device_types = ["mobile_android", "mobile_ios", "desktop_windows", "desktop_mac", "unknown"]
    dt_weights = [0.60, 0.25, 0.10, 0.03, 0.02]

    ip_countries = ["IN", "IN", "IN", "IN", "US", "AE", "SG", "GB"]

    chosen_pm = rng.choice(payment_methods, size=n_transactions, p=pm_weights)
    chosen_cn = rng.choice(card_networks, size=n_transactions, p=cn_weights)
    chosen_mc = rng.choice(merchant_categories, size=n_transactions, p=mc_weights)
    chosen_dt = rng.choice(device_types, size=n_transactions, p=dt_weights)
    chosen_ip = rng.choice(ip_countries, size=n_transactions)

    # 3. Generate amounts & assign chargebacks based on risk factors
    amounts = []
    is_3ds_list = []
    is_chargeback_list = []

    # Track customer past transaction times for velocity bursts
    customer_last_times: Dict[str, list] = {cid: [] for cid in customer_ids}

    # Baseline probability calculation
    for i in range(n_transactions):
        cid = chosen_customers[i]
        t_time = timestamps[i]
        c_mean, c_std = customer_profiles[cid]

        # Check velocity in last 1 hour
        recent_times = [t for t in customer_last_times[cid] if (t_time - t).total_seconds() <= 3600]
        customer_last_times[cid].append(t_time)
        velocity_1h = len(recent_times)

        # Decide if this transaction is an anomalous event
        is_attack = rng.random() < target_chargeback_rate

        if is_attack:
            # Anomaly: High amount spike, risky category, maybe foreign IP or no 3DS
            spike_multiplier = rng.uniform(3.5, 9.0)
            txn_amount = max(100.0, float(c_mean * spike_multiplier + rng.normal(0, c_std)))
            has_3ds = rng.choice([True, False], p=[0.25, 0.75])
            # Higher probability of chargeback
            chargeback = 1 if rng.random() < 0.85 else 0
        else:
            # Legitimate behavior: Normal spending around baseline
            txn_amount = max(50.0, float(c_mean + rng.normal(0, c_std)))
            has_3ds = rng.choice([True, False], p=[0.92, 0.08])
            # Tiny residual dispute probability (e.g. buyer regret or merchant error)
            chargeback = 1 if rng.random() < 0.003 else 0

        amounts.append(round(txn_amount, 2))
        is_3ds_list.append(bool(has_3ds))
        is_chargeback_list.append(int(chargeback))

    df = pd.DataFrame({
        "transaction_id": [f"TXN_{i+1:06d}" for i in range(n_transactions)],
        "timestamp": timestamps,
        "customer_id": chosen_customers,
        "amount": amounts,
        "payment_method": chosen_pm,
        "card_network": chosen_cn,
        "merchant_category": chosen_mc,
        "device_type": chosen_dt,
        "ip_country": chosen_ip,
        "is_3ds_authenticated": is_3ds_list,
        "is_chargeback": is_chargeback_list,
    })

    # Sort strictly chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_dataset(
    data_dir: str | Path = "data",
    force_regenerate: bool = False,
    n_transactions: int = 40000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load transaction data from data/raw or generate benchmark if not present."""
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    sample_dir = data_path / "sample"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    csv_file = raw_dir / "transactions.csv"

    if csv_file.exists() and not force_regenerate:
        logger.info("Loading existing transactions from %s", csv_file)
        df = pd.read_csv(csv_file, parse_dates=["timestamp"])
        return df

    logger.info("Generating benchmark dataset of %d transactions...", n_transactions)
    df = generate_benchmark_transactions(
        n_transactions=n_transactions,
        random_seed=random_seed,
    )

    # Save to data/raw/
    df.to_csv(csv_file, index=False)
    # Save a lightweight sample for rapid testing / UI demo
    df.head(1000).to_csv(sample_dir / "transactions_sample.csv", index=False)

    logger.info("Dataset generated and saved to %s (and sample saved).", csv_file)
    return df


def inspect_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dataset statistics, target class imbalance, and schema."""
    total_rows = len(df)
    chargeback_count = int(df["is_chargeback"].sum())
    chargeback_rate = float(df["is_chargeback"].mean())
    unique_customers = int(df["customer_id"].nunique())
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()

    missing = df.isnull().sum().to_dict()

    summary = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "unique_customers": unique_customers,
        "chargeback_count": chargeback_count,
        "chargeback_rate": chargeback_rate,
        "chargeback_percentage": f"{chargeback_rate * 100:.2f}%",
        "time_range": (str(min_time), str(max_time)),
        "missing_values": missing,
        "columns": list(df.columns),
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_dataset()
    info = inspect_dataset_summary(data)
    print("--- RiskForge AI Dataset Summary ---")
    for k, v in info.items():
        print(f"{k}: {v}")
