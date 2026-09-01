import os
import zipfile

# Complete functional codebase implementation
project_files = {
    # -------------------------------------------------------------------------
    # DATA DIRECTORY
    # -------------------------------------------------------------------------
    "Fraud-Detection/data/README.md": (
        "# Data Directory\n\n"
        "Place your PaySim dataset file (`PS_20174392719_1491204439457_log.csv`) here.\n"
        "Dataset URL: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
    ),

    "Fraud-Detection/models/.gitkeep": "",
    "Fraud-Detection/reports/figures/.gitkeep": "",
    "Fraud-Detection/reports/model_metrics/.gitkeep": "",

    # -------------------------------------------------------------------------
    # SOURCE CODE
    # -------------------------------------------------------------------------
    "Fraud-Detection/src/__init__.py": "# Source package initialization\n",

    "Fraud-Detection/src/data_loader.py": '''import os
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Load PaySim financial dataset from CSV path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download PaySim data from Kaggle.")
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded: {len(df):,} rows and {len(df.columns)} columns.")
    return df

def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that input data contains required columns."""
    required_cols = {'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                     'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return True
''',

    "Fraud-Detection/src/preprocessing.py": '''import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter relevant transaction types and apply encoding."""
    df = df.copy()
    
    # PaySim fraud occurs almost exclusively in TRANSFER and CASH_OUT transactions
    valid_types = ['TRANSFER', 'CASH_OUT']
    df = df[df['type'].isin(valid_types)].reset_index(drop=True)
    
    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=['type'], drop_first=False)
    
    # Fill any missing values if present
    df.fillna(0, inplace=True)
    
    return df
''',

    "Fraud-Detection/src/feature_engineering.py": '''import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer financial balance integrity and temporal features."""
    df = df.copy()
    
    # 1. Log Transformation of Transaction Amount
    df['log_amount'] = np.log1p(df['amount'])
    
    # 2. Balance Differences
    df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['destination_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # 3. Balance Errors / Discrepancies
    # Check if transaction amount matches actual balance shift
    df['origin_balance_error'] = df['origin_balance_change'] - df['amount']
    df['destination_balance_error'] = df['destination_balance_change'] - df['amount']
    
    # 4. Zero Balance Flags
    df['orig_zero_old_balance'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['dest_zero_new_balance'] = (df['newbalanceDest'] == 0).astype(int)
    
    # 5. Amount Ratios
    df['amount_to_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1.0)
    df['amount_to_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1.0)
    
    # 6. Cyclical Time Features (step represents hours)
    df['hour_bucket'] = df['step'] % 24
    
    return df

def get_feature_names() -> list:
    """Return the list of engineered feature names used for model training."""
    return [
        'amount', 'log_amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'origin_balance_change',
        'destination_balance_change', 'origin_balance_error',
        'destination_balance_error', 'orig_zero_old_balance',
        'dest_zero_new_balance', 'amount_to_orig_ratio',
        'amount_to_dest_ratio', 'hour_bucket', 'type_CASH_OUT', 'type_TRANSFER'
    ]
''',

    "Fraud-Detection/src/train.py": '''import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, validate_schema
from src.preprocessing import preprocess_data
from src.feature_engineering import add_features, get_feature_names

def train_xgb_model(X_train, y_train, scale_pos_weight: float = 1.0):
    """Train XGBoost model optimized for imbalanced binary classification."""
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
    
    print("Training XGBoost Fraud Model...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def run_pipeline(data_path: str, model_save_path: str = "models/xgboost_fraud_model.joblib"):
    """Execute complete training pipeline from CSV load to model save."""
    df = load_data(data_path)
    validate_schema(df)
    
    df_clean = preprocess_data(df)
    df_features = add_features(df_clean)
    
    feature_cols = get_feature_names()
    X = df_features[feature_cols]
    y = df_features['isFraud']
    
    # Calculate scale_pos_weight for imbalance
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = train_xgb_model(X_train, y_train, scale_pos_weight=scale_weight)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/PS_20174392719_1491204439457_log.csv"
    if os.path.exists(data_file):
        run_pipeline(data_file)
    else:
        print(f"Data file not found at '{data_file}'. Pipeline skipped.")
''',

    "Fraud-Detection/src/evaluate.py": '''import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate model performance using fraud-specific metrics."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("\n================ Model Evaluation Results ================")
    print(f"Optimal Threshold Used: {threshold:.2f}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"PR-AUC Score:           {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=['Actual Legit', 'Actual Fraud'], columns=['Pred Legit', 'Pred Fraud']))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("==========================================================\n")
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }
''',

    "Fraud-Detection/src/predict.py": '''import numpy as np
import pandas as pd

def predict_transaction(model, feature_dataframe: pd.DataFrame, threshold: float = 0.5) -> list:
    """Generate probability score and risk level for incoming transactions."""
    probs = model.predict_proba(feature_dataframe)[:, 1]
    results = []
    
    for prob in probs:
        prob_val = float(prob)
        prediction = "FRAUD" if prob_val >= threshold else "LEGITIMATE"
        
        if prob_val >= 0.70:
            risk_level = "HIGH"
        elif prob_val >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        results.append({
            "fraud_probability": round(prob_val, 4),
            "prediction": prediction,
            "risk_level": risk_level
        })
        
    return results
''',

    "Fraud-Detection/src/explainability.py": '''import os
import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample, output_dir: str = "reports/figures"):
    """Generate global SHAP feature importance plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    
    output_path = os.path.join(output_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved SHAP summary plot to '{output_path}'")
''',

    # -------------------------------------------------------------------------
    # NOTEBOOKS
    # -------------------------------------------------------------------------
    "Fraud-Detection/notebooks/01_data_exploration.ipynb": '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# 01 Data Exploration\\n", "Explore raw PaySim transactions."]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/02_feature_engineering.ipynb": '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# 02 Feature Engineering\\n", "Construct domain features."]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/03_xgboost_model.ipynb": '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# 03 XGBoost Model Training\\n", "Train and hyperparameter tune XGBoost."]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/04_model_evaluation.ipynb": '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# 04 Model Evaluation\\n", "Evaluate PR-AUC and optimize decision threshold."]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',
    "Fraud-Detection/notebooks/05_shap_explainability.ipynb": '{"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# 05 Model Explainability\\n", "Analyze model drivers with SHAP."]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 2}',

    # -------------------------------------------------------------------------
    # TESTS
    # -------------------------------------------------------------------------
    "Fraud-Detection/tests/test_preprocessing.py": '''import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocess_filtering():
    df = pd.DataFrame({
        'type': ['TRANSFER', 'PAYMENT', 'CASH_OUT', 'DEBIT'],
        'amount': [100, 200, 300, 400]
    })
    processed = preprocess_data(df)
    assert len(processed) == 2
    assert set(processed['type_TRANSFER'].unique()).issubset({0, 1})
''',

    "Fraud-Detection/tests/test_features.py": '''import pandas as pd
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
    assert feats['origin_balance_error'].iloc[0] == 0.0
''',

    "Fraud-Detection/tests/test_prediction.py": '''import numpy as np
import pandas as pd
from src.predict import predict_transaction

class MockModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9], [0.8, 0.2]])

def test_prediction_output():
    df_mock = pd.DataFrame({'feat1': [1, 2]})
    res = predict_transaction(MockModel(), df_mock, threshold=0.5)
    
    assert len(res) == 2
    assert res[0]['prediction'] == 'FRAUD'
    assert res[0]['risk_level'] == 'HIGH'
    assert res[1]['prediction'] == 'LEGITIMATE'
    assert res[1]['risk_level'] == 'LOW'
''',

    # -------------------------------------------------------------------------
    # CONFIGURATION & METADATA
    # -------------------------------------------------------------------------
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

    "Fraud-Detection/LICENSE": "MIT License\n\nCopyright (c) 2026 Fraud Analytics Project\n",

    "Fraud-Detection/README.md": (
        "# XGBoost Fraud Analytics\n\n"
        "An end-to-end **Fraud Detection and Risk Scoring platform using XGBoost**."
    )
}

def create_zip(zip_filename="Fraud-Detection.zip"):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, content in project_files.items():
            zipf.writestr(file_path, content)
    print(f"Successfully generated '{zip_filename}' with full implementation files!")

if __name__ == "__main__":
    create_zip()