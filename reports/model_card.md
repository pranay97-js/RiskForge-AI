# RiskForge AI Model Card

## Model Details
- **Developer:** RiskForge AI Engineering Team
- **Model Release Date:** September 2026
- **Model Version:** v1.0.0 (Production Submission)
- **Model Architecture:** Gradient Boosted Decision Trees (`XGBClassifier`) with calibrated probability estimation, compared against an L2-regularized `LogisticRegression` baseline.
- **Explainability Engine:** TreeSHAP (`shap.TreeExplainer`) for fast per-prediction attribution.
- **Verification Engine:** Deterministic Rule Registry (6 auditable rule checks).
- **Decision Engine:** Deterministic Multi-Signal Fusion Matrix with Cost Optimization.
- **License:** Proprietary / Enterprise Defense-Only

---

## Intended Use
- **Primary Use Case:** Prioritizing e-commerce and payment gateway transactions exhibiting elevated dispute and chargeback risk.
- **Target Loss Class:** Transaction chargebacks, payment disputes, and friendly fraud.
- **Target Users:** Merchant risk analysts, payment support teams, and merchant business owners.
- **Out of Scope / Prohibited Use:** Offensive fraud generation, payment bypass testing, credential theft, attacker simulation, or denial-of-service evasion tooling.

---

## Training Data & Methodology
- **Benchmark Source:** High-fidelity multi-entity synthetic benchmark dataset structured for payment gateway risk operations (inspired by IEEE-CIS and payment gateway threat vectors; not proprietary live Razorpay production customer records).
- **Target Definition:** `is_chargeback` (Binary: 1 for confirmed chargeback dispute, 0 for legitimate settled transaction). Used as a proxy for merchant chargeback liability.
- **Target Imbalance:** ~2.45% positive dispute rate (reflecting authentic payment gateway environments).
- **Temporal Splitting Strategy:** Strict chronological Out-Of-Time (OOT) partitioning:
  - **Train (70%):** 28,000 transactions (2026-01-01 to 2026-02-12).
  - **Validation (15%):** 6,000 transactions (2026-02-12 to 2026-02-21).
  - **Frozen Final Test (15%):** 6,000 transactions (2026-02-21 to 2026-03-02) evaluated once in Day 4.
- **Leakage Safeguards:** Stateful customer baselines (mean, variance, velocity counters) maintain continuous expanding windows across splits without lookahead leakage.

---

## Model Features (43 Total)
1. **Transaction Context:** `amount`, `log_amount`, `hour_of_day`, `day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`, `is_3ds_authenticated`.
2. **Velocity Signals:** `velocity_1h`, `velocity_6h`, `velocity_24h`.
3. **Customer Baselines:** `customer_prior_txn_count`, `customer_historical_mean`, `customer_historical_std`, `is_first_transaction`.
4. **Behavioral Deviation Scores (Section 4.3):**
   - Amount Ratio: $\frac{\text{current\_amount}}{\text{customer\_historical\_avg} + \epsilon}$
   - Amount Z-Score: $\frac{\text{current\_amount} - \text{customer\_mean}}{\text{customer\_std} + \epsilon}$
   - Velocity Ratio: $\frac{\text{velocity\_1h}}{\text{hourly\_baseline} + 0.1}$
5. **Categoricals:** One-hot encoded `payment_method`, `card_network`, `merchant_category`, `device_type`, `ip_country`.

---

## Performance Metrics (Frozen Held-Out Test Partition)
- **PR-AUC (XGBoost Final Candidate):** `0.7331` (vs. `0.4279` for Baseline Logistic Regression).
- **ROC-AUC (XGBoost Final Candidate):** `0.9327`.
- **False-Positive Reduction:** Eliminated **59.1% of false investigations** compared to the baseline linear model (45 vs 110 FPs).
- **Cost-Optimal Cutoff:** $\tau^* = 0.200$ (minimizing net operational dispute cost).

---

## Limitations & Edge Cases
1. **Cold-Start Entities:** New customers have no prior history; the system applies Bayesian fallback priors with an explicit `is_first_transaction` flag and assigns `DATA_INSUFFICIENT` if context is absent.
2. **Dispute Settlement Lag:** Real-world chargeback labels take 30–90 days to settle. In production environments, sliding label lag windows must be enforced.
3. **Synthetic Proxy Data:** Initial development uses high-fidelity synthetic proxy data (no actual Razorpay schemas or production data are used); performance on proprietary live production streams must be calibrated per merchant.

---

## Ethical Considerations & Safety
- **Defense-Only Posture:** The system contains zero offensive fraud execution tools.
- **Privacy Preservation:** Customer PII (names, PAN, CVV) is excluded; only masked entity identifiers and statistical aggregates are stored or processed.
- **Deterministic AI Grounding:** The LLM is strictly an operational assistant and cannot override the deterministic risk engine or invent transaction evidence.
