# RiskForge AI — Day 1 Experiment Log & Benchmark

**Execution Timestamp:** 2026-09-05 21:16:16  
**Target Variable:** `is_chargeback` (Binary 0/1)  
**Evaluation Strategy:** Strict Chronological Out-of-Time (70% Train, 15% Validation, 15% Frozen Final Test)  

---

## 1. Dataset & Split Summary

| Partition | Row Count | Start Time | End Time | Chargebacks | Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (70%)** | 28,000 | 2026-01-01 00:00:13.574571275 | 2026-02-12 08:10:40.268238776 | 694 | 2.48% |
| **Val (15%)** | 6,000 | 2026-02-12 08:13:28.337242732 | 2026-02-21 10:57:29.365429724 | 145 | 2.42% |
| **Frozen Test (15%)** | 6,000 | 2026-02-21 11:02:32.031124202 | 2026-03-02 10:35:07.833662134 | 147 | 2.45% |

*Note: In compliance with buildathon rules, the Frozen Test partition remains untouched and will only be evaluated once during Day 4.*

---

## 2. Feature Pipeline Summary

- **Total Extracted Features:** 43
- **Feature Categories:**
  - *Contextual & Temporal:* `amount`, `log_amount`, `hour_of_day`, `day_of_week`, `hour_sin`, `hour_cos`, `is_weekend`, `is_3ds_authenticated`
  - *Velocity Tracking:* `velocity_1h`, `velocity_6h`, `velocity_24h`
  - *Customer Behavioral Baseline:* `customer_prior_txn_count`, `customer_historical_mean`, `customer_historical_std`, `is_first_transaction`
  - *Behavioral Deviation Scores:* `amount_ratio`, `amount_zscore`, `velocity_ratio`
  - *Categorical Encodings:* One-hot encoded `payment_method`, `card_network`, `merchant_category`, `device_type`, `ip_country`

---

## 3. Validation Model Comparison (Cutoff = 0.50)

| System | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 41.92% | 66.21% | 0.5134 | 0.5717 | 0.7770 | 0.0407 |
| **XGBoost (Final Candidate)** | **66.49%** | **87.59%** | **0.7560** | **0.6953** | **0.9327** | **0.0166** |

---

## 4. Confusion Matrix Analysis (Validation Set)

### Baseline (Logistic Regression)
```
--------------------------------------------------
                PREDICTED: 0        PREDICTED: 1  
ACTUAL: 0       TN: 5722            FP: 133           
ACTUAL: 1       FN: 49              TP: 96            
--------------------------------------------------
False Positive Rate (FPR): 2.27%
False Discovery Rate (FDR): 58.08%

```

### Candidate (XGBoost)
```
--------------------------------------------------
                PREDICTED: 0        PREDICTED: 1  
ACTUAL: 0       TN: 5791            FP: 64            
ACTUAL: 1       FN: 18              TP: 127           
--------------------------------------------------
False Positive Rate (FPR): 1.09%
False Discovery Rate (FDR): 33.51%

```

---

## 5. Day 1 Key Conclusions & Definition of Done
- **Dataset Understood & Controlled:** Successfully generated and inspected high-fidelity multi-entity transactions with ~2.5% realistic chargeback imbalance.
- **Zero Lookahead Leakage:** Customer spending averages, standard deviations, and velocity counts are calculated strictly on transactions occurring prior to the target event.
- **Model Supremacy:** XGBoost demonstrates substantial superiority over the linear baseline on imbalanced tabular data, achieving a high PR-AUC and balanced precision/recall.
- **Reproducibility:** All artifacts (`models/baseline.pkl`, `models/xgboost.pkl`, `models/feature_pipeline.pkl`, `models/metadata.json`) and frozen test partition (`data/processed/test_frozen.csv`) are safely committed and reproducible.
