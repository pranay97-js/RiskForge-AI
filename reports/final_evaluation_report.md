# RiskForge AI — Final Held-Out Evaluation Report
> **Evaluation Protocol:** Frozen Held-Out Out-Of-Time Test Set (Section 10.3)  
> **Evaluation Timestamp:** 2026-09-05 21:16:36  
> **Dataset Partition:** `data/processed/test_frozen.csv`  
> **Total Evaluated Records:** 6,000 transactions  
> **Total Chargeback Disputes:** 147 (2.45% positive rate)  

---

## 1. Official Results Table (Section 11 Specification)

All figures in this table represent **actual measured values** evaluated on the frozen test set:

| System | Precision | Recall | F1-Score | PR-AUC | Expected Operational Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline, $\tau=0.50$)** | 42.41% | 55.10% | 0.4793 | 0.4279 | ₹4,467,084.86 |
| **XGBoost (Candidate, $\tau=0.50$)** | 74.14% | 87.76% | 0.8037 | 0.7331 | ₹1,960,622.69 |
| **XGBoost (Cost-Optimal, $\tau^*=0.200$)** | 44.33% | 87.76% | 0.5890 | 0.7331 | ₹1,767,170.25 |
| **Final Operational Policy (ML + Rules + Fusion)** | **42.90%** | **88.44%** | **0.5778** | **0.7331** | **₹1,874,400.28** |

---

## 2. Business Impact & False-Positive Burden Analysis

- **Total Baseline Exposure (Unchecked):** ₹6,474,066.47
- **Baseline Logistic Regression False Positives:** 110 (FPR: 1.88%)
- **XGBoost Candidate False Positives:** 45 (FPR: 0.77%)
- **False-Positive Reduction:** XGBoost eliminates **65 false investigations** (a 59.1% reduction).
- **Net Avoidable Merchant Savings:** The Final Operational Policy recovers **₹2,675,314.87** in preventable dispute losses net of labor investigation costs.

---

## 3. Confusion Matrix Breakdown (Frozen Test Set)

### Baseline Logistic Regression
```
--------------------------------------------------
                PREDICTED: 0        PREDICTED: 1  
ACTUAL: 0       TN: 5743            FP: 110           
ACTUAL: 1       FN: 66              TP: 81            
--------------------------------------------------
False Positive Rate (FPR): 1.88%
False Discovery Rate (FDR): 57.59%

```

### Final Candidate XGBoost (Default Cutoff $\tau = 0.50$)
```
--------------------------------------------------
                PREDICTED: 0        PREDICTED: 1  
ACTUAL: 0       TN: 5808            FP: 45            
ACTUAL: 1       FN: 18              TP: 129           
--------------------------------------------------
False Positive Rate (FPR): 0.77%
False Discovery Rate (FDR): 25.86%

```

### Final Operational Fusion Policy
```
--------------------------------------------------
                PREDICTED: 0        PREDICTED: 1  
ACTUAL: 0       TN: 5680            FP: 173           
ACTUAL: 1       FN: 17              TP: 130           
--------------------------------------------------
False Positive Rate (FPR): 2.96%
False Discovery Rate (FDR): 57.10%

```

---

## 4. Frozen Evaluation Protocol Verification
- **One-Shot Execution:** The model, feature engineering pipeline, deterministic rules, and decision fusion thresholds were finalized during Days 1–3 and frozen prior to running this script.
- **Zero Leakage:** No hyperparameter tuning, rule modifications, or feature selections were performed against this partition.
- **Visual Evidence:** High-resolution figures exported to `reports/figures/pr_curve_comparison.png` and `reports/figures/roc_curve_comparison.png`.
