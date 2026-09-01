import os
import zipfile

# Define folder structure and file contents
project_files = {
    # Data directory placeholder
    "Fraud-Detection/data/README.md": (
        "# Data Directory\n\n"
        "Place your PaySim dataset file (`PS_20174392719_1491204439457_log.csv`) here.\n"
        "Dataset URL: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
    ),

    # Models directory placeholder
    "Fraud-Detection/models/.gitkeep": "",

    # Reports directories
    "Fraud-Detection/reports/figures/.gitkeep": "",
    "Fraud-Detection/reports/model_metrics/.gitkeep": "",

    # Source code files
    "Fraud-Detection/src/__init__.py": "# Source package initialization\n",

    "Fraud-Detection/src/data_loader.py": '''import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """Load PaySim financial dataset from CSV path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download PaySim data.")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df
''',

    "Fraud-Detection/src/preprocessing.py": '''import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, check nulls, and format transaction types."""
    df = df.copy()
    # Filter for transaction types where fraud typically occurs (TRANSFER & CASH_OUT)
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].reset_index(drop=True)
    
    # One-hot encoding for transaction type
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    return df
''',

    "Fraud-Detection/src/feature_engineering.py": '''import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific financial fraud features."""
    df = df.copy()
    
    # Log transformation of amount
    df['log_amount'] = np.log1p(df['amount'])
    
    # Balance changes
    df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['destination_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Balance discrepancy checks
    df['origin_balance_error'] = df['origin_balance_change'] - df['amount']
    df['destination_balance_error'] = df['destination_balance_change'] - df['amount']
    
    # Amount ratios
    df['amount_to_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    
    # Temporal step features
    df['hour_bucket'] = df['step'] % 24
    
    return df
''',

    "Fraud-Detection/src/train.py": '''import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def train_xgb(X_train, y_train, scale_pos_weight=1.0):
    """Train XGBoost Classifier optimized for imbalanced fraud data."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("XGBoost Fraud Model Training Script Initialized.")
''',

    "Fraud-Detection/src/evaluate.py": '''import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance with fraud-specific metrics."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "confusion_matrix": cm}
''',

    "Fraud-Detection/src/predict.py": '''import json
import numpy as np

def predict_transaction(model, feature_vector, threshold=0.5):
    """Make real-time risk predictions for a single transaction vector."""
    prob = float(model.predict_proba(feature_vector)[:, 1][0])
    prediction = "FRAUD" if prob >= threshold else "LEGITIMATE"
    
    if prob >= 0.70:
        risk_level = "HIGH"
    elif prob >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
        
    return {
        "fraud_probability": round(prob, 4),
        "prediction": prediction,
        "risk_level": risk_level
    }
''',

    "Fraud-Detection/src/explainability.py": '''import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample):
    """Generate global SHAP explainability plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png")
    plt.close()
    print("Saved SHAP summary plot to reports/figures/shap_summary.png")
''',

    # Jupyter Notebook Placeholders
    "Fraud-Detection/notebooks/01_data_exploration.ipynb": '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/02_feature_engineering.ipynb": '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/03_xgboost_model.ipynb": '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/04_model_evaluation.ipynb": '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/05_shap_explainability.ipynb": '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',

    # Unit Tests
    "Fraud-Detection/tests/test_preprocessing.py": '''import pytest
import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocess_filtering():
    df = pd.DataFrame({
        'type': ['TRANSFER', 'PAYMENT', 'CASH_OUT', 'DEBIT'],
        'amount': [100, 200, 300, 400]
    })
    processed = preprocess_data(df)
    assert len(processed) == 2
''',

    "Fraud-Detection/tests/test_features.py": '''import pytest
import pandas as pd
from src.feature_engineering import add_features

def test_feature_creation():
    df = pd.DataFrame({
        'amount': [100.0],
        'oldbalanceOrg': [500.0],
        'newbalanceOrig': [400.0],
        'oldbalanceDest': [0.0],
        'newbalanceDest': [100.0],
        'step': [1]
    })
    feats = add_features(df)
    assert 'log_amount' in feats.columns
    assert feats['origin_balance_change'].iloc[0] == 100.0
''',

    "Fraud-Detection/tests/test_prediction.py": '''import pytest
import numpy as np
from src.predict import predict_transaction

class MockModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])

def test_prediction_output():
    res = predict_transaction(MockModel(), [[0]*10], threshold=0.5)
    assert res['prediction'] == 'FRAUD'
    assert res['risk_level'] == 'HIGH'
''',

    # Config & Metadata Files
    "Fraud-Detection/requirements.txt": (
        "pandas>=1.5.0\n"
        "numpy>=1.21.0\n"
        "scikit-learn>=1.1.0\n"
        "xgboost>=1.6.0\n"
        "shap>=0.41.0\n"
        "matplotlib>=3.5.0\n"
        "seaborn>=0.11.0\n"
        "joblib>=1.1.0\n"
        "pytest>=7.0.0\n"
    ),

    "Fraud-Detection/.gitignore": (
        "# Python & Virtualenv\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        ".venv/\n"
        "venv/\n"
        "env/\n\n"
        "# Data files\n"
        "data/*.csv\n"
        "data/*.zip\n\n"
        "# Models\n"
        "models/*.pkl\n"
        "models/*.json\n"
        "models/*.joblib\n\n"
        "# IDE & OS\n"
        ".vscode/\n"
        ".idea/\n"
        ".DS_Store\n"
    ),

    "Fraud-Detection/LICENSE": "MIT License\n\nCopyright (c) 2026 Fraud Analytics Project\n\nPermission is hereby granted...",

    # README Markdown
    "Fraud-Detection/README.md": '''# XGBoost Fraud Analytics

An end-to-end **Fraud Detection and Risk Scoring platform using XGBoost**, developed using the PaySim synthetic financial transaction dataset.

The project focuses on building a production-oriented machine learning workflow for identifying potentially fraudulent financial transactions while addressing the key challenges associated with fraud analytics, including **high class imbalance, feature engineering, model explainability, threshold optimization, and false-positive management**.

---

## Project Overview

Fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a small fraction of total transactions.

This project uses **XGBoost** to develop a fraud classification model capable of assigning a fraud probability to individual financial transactions.

The solution covers the complete machine learning lifecycle:

```text
Raw Transaction Data
        |
        v
Data Quality & Exploration
        |
        v
Feature Engineering
        |
        v
Train / Validation / Test Split
        |
        v
XGBoost Fraud Classifier
        |
        v
Probability Scoring
        |
        v
Threshold Optimization
        |
        v
Fraud / Non-Fraud Decision
        |
        v
Model Explainability & Analytics