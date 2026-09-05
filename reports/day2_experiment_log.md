# RiskForge AI — Day 2 Experiment Log: Risk Engine & Verification

**Execution Date:** 2026-09-05  
**Focus:** Cost Engine, Deterministic Rules, Decision Fusion, and SHAP Explainability  

---

## 1. Cost-Sensitive Threshold Optimization (Validation Set)

Evaluated on 6,000 chronological validation transactions (145 chargebacks, 2.42% positive rate):
- **Investigation Cost:** ₹50.00 / review
- **Chargeback Assessment Penalty:** ₹3,000.00
- **Preventable Loss Rate:** 90%

| Policy Cutoff | Precision | Recall | Review Rate | Operational Cost (INR) |
| :--- | :--- | :--- | :--- | :--- |
| **Default ($	au = 0.500$)** | 43.61% | 80.00% | 4.43% | ₹396,722.08 |
| **Cost-Optimal ($	au = 0.200$)** | **21.17%** | **84.83%** | **9.68%** | **₹360,942.29** |
| **Net Operational Benefit** | — | — | — | **+₹35,779.79 Saved** |

---

## 2. Deterministic Rule Registry Status

All 6 deterministic rules implemented, versioned, and verified:
1. `VELOCITY_ANOMALY` (Behavioral, Severity: HIGH) — Catches high order frequency spikes (>3.0x).
2. `SPENDING_DEVIATION` (Behavioral, Severity: HIGH) — Flags amounts > 3.5x historical baseline or z-score > 3.5.
3. `DISPUTE_HISTORY` (History, Severity: CRITICAL) — Flags prior dispute/chargeback records.
4. `NEW_ACCOUNT_HIGH_VALUE` (History, Severity: MEDIUM) — Flags first-time purchases exceeding 90th percentile value.
5. `PAYMENT_CHANGE` (Consistency, Severity: MEDIUM) — Flags non-3DS cross-border or high-value payment shifts.
6. `DATA_INSUFFICIENT` (Data Quality, Severity: LOW) — Safeguard forcing `REVIEW` when essential fields are absent.

---

## 3. Decision Fusion Engine Architecture

Resolves multi-signal operational states:
$$\text{ML Probability} + \text{Rule Severity} + \text{Data Quality} + \text{Financial Exposure} \Longrightarrow \text{Final State}$$

- **Convergence State (`HIGH`):** Both ML and Rules trigger with acceptable data quality.
- **Convergence State (`LOW`):** Both ML and Rules indicate legitimate baseline behavior.
- **Disagreement Fallback (`REVIEW`):** Disagreement between statistical detector and deterministic rules triggers human review.
- **Data Quality Safeguard (`REVIEW`):** Poor data quality inhibits auto-approval/auto-decline.


---

## 4. SHAP Explainability Integration

Integrated `shap.TreeExplainer` on the XGBoost candidate model:
- Exposes per-transaction top positive (risk-elevating) and negative (protective) factors.
- Provides grounded, verifiable attribution inputs for the subsequent LLM operational layer.
