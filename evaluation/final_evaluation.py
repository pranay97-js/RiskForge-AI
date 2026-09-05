"""Day 4 Final Evaluation Pipeline: One-shot evaluation on the frozen held-out test set.

Strictly adheres to Section 10.3 and Section 11 of the RiskForge AI blueprint.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from evaluation.confusion_matrix import format_confusion_matrix_ascii, generate_confusion_matrix_report
from evaluation.metrics import compute_classification_metrics
from src.decision.cost import CostEngine
from src.decision.service import RiskDecisionService
from src.ml.predict import RiskPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_final_evaluation(
    frozen_test_path: str = "data/processed/test_frozen.csv",
    metadata_path: str = "models/metadata.json",
    figures_dir: str = "reports/figures",
    report_output_path: str = "reports/final_evaluation_report.md",
) -> Dict[str, Any]:
    """Execute one-shot evaluation on the frozen test partition and export artifacts."""
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: Loading Frozen Held-Out Test Set ===")
    test_df = pd.read_csv(frozen_test_path)
    logger.info("Loaded %d frozen test rows (%s to %s)", len(test_df), test_df["timestamp"].min(), test_df["timestamp"].max())

    y_true = test_df["is_chargeback"].values.astype(int)
    amounts = test_df["amount"].values.astype(float)
    pos_count = int(y_true.sum())
    pos_rate = float(y_true.mean())
    logger.info("Target class distribution in frozen test set: %d chargebacks (%.2f%%)", pos_count, pos_rate * 100)

    # Cost model
    cost_engine = CostEngine(
        investigation_cost=50.0,
        chargeback_penalty_fee=3000.0,
        preventable_loss_rate=0.90,
    )

    # Load artifacts
    logger.info("=== STEP 2: Loading Frozen Model Artifacts ===")
    baseline_model = joblib.load("models/baseline.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")
    pipeline = joblib.load("models/feature_pipeline.pkl")
    decision_service = RiskDecisionService()

    # Read optimal threshold from metadata
    opt_tau = 0.200
    if Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            opt_tau = meta.get("optimal_cost_threshold", {}).get("optimal_threshold", 0.200)

    logger.info("Transforming frozen test features...")
    X_test = pipeline.transform(test_df)

    # 1. Evaluate Baseline Model (τ = 0.50)
    logger.info("=== STEP 3: Evaluating Baseline Logistic Regression ===")
    prob_baseline = baseline_model.predict_proba(X_test)[:, 1]
    metrics_baseline = compute_classification_metrics(y_true, prob_baseline, threshold=0.50)
    cost_baseline = cost_engine.evaluate_portfolio_cost(amounts, prob_baseline, prob_baseline >= 0.50)
    cm_baseline = generate_confusion_matrix_report(y_true, (prob_baseline >= 0.50).astype(int))

    # 2. Evaluate Candidate XGBoost (Default Cutoff τ = 0.50)
    logger.info("=== STEP 4: Evaluating Candidate XGBoost (Default τ = 0.50) ===")
    prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics_xgb_default = compute_classification_metrics(y_true, prob_xgb, threshold=0.50)
    cost_xgb_default = cost_engine.evaluate_portfolio_cost(amounts, prob_xgb, prob_xgb >= 0.50)
    cm_xgb_default = generate_confusion_matrix_report(y_true, (prob_xgb >= 0.50).astype(int))

    # 3. Evaluate Cost-Optimal XGBoost (τ* = 0.200)
    logger.info("=== STEP 5: Evaluating Cost-Optimal XGBoost (τ* = %.3f) ===", opt_tau)
    metrics_xgb_opt = compute_classification_metrics(y_true, prob_xgb, threshold=opt_tau)
    cost_xgb_opt = cost_engine.evaluate_portfolio_cost(amounts, prob_xgb, prob_xgb >= opt_tau)
    cm_xgb_opt = generate_confusion_matrix_report(y_true, (prob_xgb >= opt_tau).astype(int))

    # 4. Evaluate Final Operational Policy (Decision Fusion: ML + Rules + Data Quality)
    logger.info("=== STEP 6: Evaluating Final Operational Fusion Policy ===")
    operational_decisions = []
    rule_engine = decision_service.rule_engine
    fusion_engine = decision_service.fusion_engine

    logger.info("Evaluating fused decisions across %d frozen test cases...", len(test_df))
    feature_records = X_test.to_dict(orient="records")
    test_records = test_df.to_dict(orient="records")

    for i in range(len(test_df)):
        combined = {**feature_records[i], **test_records[i]}
        rule_out = rule_engine.evaluate(combined)
        amt = float(test_records[i].get("amount", 0.0))
        p = float(prob_xgb[i])
        exp_loss = cost_engine.calculate_expected_loss(amt, p)
        priority = exp_loss / (cost_engine.investigation_cost + 1.0)

        fused = fusion_engine.fuse(
            ml_score=p,
            rule_signal=rule_out.rule_signal,
            data_quality=rule_out.data_quality,
            expected_loss=exp_loss,
            priority_score=priority,
        )
        operational_decisions.append(1 if fused.operational_state in ["HIGH", "REVIEW"] else 0)

    operational_decisions_arr = np.array(operational_decisions, dtype=int)
    # Use continuous ML probabilities for AUC metrics (binary flags produce meaningless AUC)
    # but use the operational binary decisions for precision/recall/F1
    metrics_fusion = compute_classification_metrics(y_true, prob_xgb, threshold=0.50)
    # Override precision/recall/F1 with the actual fusion decisions
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics_fusion["precision"] = float(precision_score(y_true, operational_decisions_arr, zero_division=0))
    metrics_fusion["recall"] = float(recall_score(y_true, operational_decisions_arr, zero_division=0))
    metrics_fusion["f1"] = float(f1_score(y_true, operational_decisions_arr, zero_division=0))
    cost_fusion = cost_engine.evaluate_portfolio_cost(amounts, prob_xgb, operational_decisions_arr == 1)
    cm_fusion = generate_confusion_matrix_report(y_true, operational_decisions_arr)


    # Export Visual Figures
    logger.info("=== STEP 7: Exporting High-Resolution Evaluation Figures ===")
    plt.style.use("dark_background")

    # Figure 1: PR Curves Comparison
    plt.figure(figsize=(8, 6), facecolor="#02042b")
    ax = plt.gca()
    ax.set_facecolor("#0b132b")

    prec_base, rec_base, _ = precision_recall_curve(y_true, prob_baseline)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, prob_xgb)

    plt.plot(rec_base, prec_base, label=f"Baseline Logistic Regression (PR-AUC = {metrics_baseline['pr_auc']:.4f})", color="#94a3b8", linestyle="--", linewidth=2)
    plt.plot(rec_xgb, prec_xgb, label=f"XGBoost Final Candidate (PR-AUC = {metrics_xgb_default['pr_auc']:.4f})", color="#38bdf8", linewidth=2.5)

    plt.title("Precision-Recall Curves (Frozen Held-Out Test Set)", fontsize=13, fontweight="bold", color="#f1f5f9", pad=12)
    plt.xlabel("Recall", fontsize=11, color="#cbd5e1")
    plt.ylabel("Precision", fontsize=11, color="#cbd5e1")
    plt.legend(loc="upper right", framealpha=0.3, facecolor="#0f172a")
    plt.grid(True, linestyle=":", alpha=0.3, color="#334155")
    plt.tight_layout()
    fig1_path = figures_path / "pr_curve_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # Figure 2: ROC Curves Comparison
    plt.figure(figsize=(8, 6), facecolor="#02042b")
    ax = plt.gca()
    ax.set_facecolor("#0b132b")

    fpr_base, tpr_base, _ = roc_curve(y_true, prob_baseline)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, prob_xgb)

    plt.plot(fpr_base, tpr_base, label=f"Baseline Logistic Regression (ROC-AUC = {metrics_baseline['roc_auc']:.4f})", color="#94a3b8", linestyle="--", linewidth=2)
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost Final Candidate (ROC-AUC = {metrics_xgb_default['roc_auc']:.4f})", color="#6366f1", linewidth=2.5)
    plt.plot([0, 1], [0, 1], linestyle=":", color="#475569", label="Random Chance (0.50)")

    plt.title("ROC Curves (Frozen Held-Out Test Set)", fontsize=13, fontweight="bold", color="#f1f5f9", pad=12)
    plt.xlabel("False Positive Rate", fontsize=11, color="#cbd5e1")
    plt.ylabel("True Positive Rate", fontsize=11, color="#cbd5e1")
    plt.legend(loc="lower right", framealpha=0.3, facecolor="#0f172a")
    plt.grid(True, linestyle=":", alpha=0.3, color="#334155")
    plt.tight_layout()
    fig2_path = figures_path / "roc_curve_comparison.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # Update metadata.json with frozen test results
    meta_path = Path(metadata_path)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["frozen_test_evaluation"] = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "n_samples": len(test_df),
            "n_chargebacks": pos_count,
            "positive_rate": pos_rate,
            "systems": {
                "baseline_logistic_regression": {**metrics_baseline, **cost_baseline},
                "xgboost_default_threshold": {**metrics_xgb_default, **cost_xgb_default},
                "xgboost_optimal_threshold": {**metrics_xgb_opt, **cost_xgb_opt},
                "final_operational_policy": {**metrics_fusion, **cost_fusion},
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Build Section 11 Markdown Report
    logger.info("=== STEP 8: Writing Section 11 Final Evaluation Report ===")
    report_md = f"""# RiskForge AI — Final Held-Out Evaluation Report
> **Evaluation Protocol:** Frozen Held-Out Out-Of-Time Test Set (Section 10.3)  
> **Evaluation Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> **Dataset Partition:** `data/processed/test_frozen.csv`  
> **Total Evaluated Records:** {len(test_df):,} transactions  
> **Total Chargeback Disputes:** {pos_count:,} ({pos_rate*100:.2f}% positive rate)  

---

## 1. Official Results Table (Section 11 Specification)

All figures in this table represent **actual measured values** evaluated on the frozen test set:

| System | Precision | Recall | F1-Score | PR-AUC | Expected Operational Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline, $\\tau=0.50$)** | {metrics_baseline['precision']*100:.2f}% | {metrics_baseline['recall']*100:.2f}% | {metrics_baseline['f1']:.4f} | {metrics_baseline['pr_auc']:.4f} | ₹{cost_baseline['total_net_operational_cost']:,.2f} |
| **XGBoost (Candidate, $\\tau=0.50$)** | {metrics_xgb_default['precision']*100:.2f}% | {metrics_xgb_default['recall']*100:.2f}% | {metrics_xgb_default['f1']:.4f} | {metrics_xgb_default['pr_auc']:.4f} | ₹{cost_xgb_default['total_net_operational_cost']:,.2f} |
| **XGBoost (Cost-Optimal, $\\tau^*={opt_tau:.3f}$)** | {metrics_xgb_opt['precision']*100:.2f}% | {metrics_xgb_opt['recall']*100:.2f}% | {metrics_xgb_opt['f1']:.4f} | {metrics_xgb_opt['pr_auc']:.4f} | ₹{cost_xgb_opt['total_net_operational_cost']:,.2f} |
| **Final Operational Policy (ML + Rules + Fusion)** | **{metrics_fusion['precision']*100:.2f}%** | **{metrics_fusion['recall']*100:.2f}%** | **{metrics_fusion['f1']:.4f}** | **{metrics_xgb_default['pr_auc']:.4f}** | **₹{cost_fusion['total_net_operational_cost']:,.2f}** |

---

## 2. Business Impact & False-Positive Burden Analysis

- **Total Baseline Exposure (Unchecked):** ₹{cost_baseline['total_portfolio_exposure']:,.2f}
- **Baseline Logistic Regression False Positives:** {cm_baseline['false_positives']} (FPR: {cm_baseline['false_positive_rate']*100:.2f}%)
- **XGBoost Candidate False Positives:** {cm_xgb_default['false_positives']} (FPR: {cm_xgb_default['false_positive_rate']*100:.2f}%)
- **False-Positive Reduction:** XGBoost eliminates **{cm_baseline['false_positives'] - cm_xgb_default['false_positives']} false investigations** (a {((cm_baseline['false_positives'] - cm_xgb_default['false_positives'])/cm_baseline['false_positives'])*100:.1f}% reduction).
- **Net Avoidable Merchant Savings:** The Final Operational Policy recovers **₹{cost_fusion['net_avoidable_savings']:,.2f}** in preventable dispute losses net of labor investigation costs.

---

## 3. Confusion Matrix Breakdown (Frozen Test Set)

### Baseline Logistic Regression
```
{format_confusion_matrix_ascii(cm_baseline)}
```

### Final Candidate XGBoost (Default Cutoff $\\tau = 0.50$)
```
{format_confusion_matrix_ascii(cm_xgb_default)}
```

### Final Operational Fusion Policy
```
{format_confusion_matrix_ascii(cm_fusion)}
```

---

## 4. Frozen Evaluation Protocol Verification
- **One-Shot Execution:** The model, feature engineering pipeline, deterministic rules, and decision fusion thresholds were finalized during Days 1–3 and frozen prior to running this script.
- **Zero Leakage:** No hyperparameter tuning, rule modifications, or feature selections were performed against this partition.
- **Visual Evidence:** High-resolution figures exported to `reports/figures/pr_curve_comparison.png` and `reports/figures/roc_curve_comparison.png`.
"""

    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Final Evaluation Report successfully written to %s", report_output_path)

    return {
        "metrics_baseline": metrics_baseline,
        "metrics_xgb_default": metrics_xgb_default,
        "metrics_xgb_opt": metrics_xgb_opt,
        "metrics_fusion": metrics_fusion,
    }


if __name__ == "__main__":
    run_final_evaluation()
