import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate model performance using fraud-specific metrics."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("
================ Model Evaluation Results ================")
    print(f"Optimal Threshold Used: {threshold:.2f}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"PR-AUC Score:           {pr_auc:.4f}")
    print("
Confusion Matrix:")
    print(pd.DataFrame(cm, index=['Actual Legit', 'Actual Fraud'], columns=['Pred Legit', 'Pred Fraud']))
    print("
Classification Report:")
    print(classification_report(y_test, preds))
    print("==========================================================
")
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }
