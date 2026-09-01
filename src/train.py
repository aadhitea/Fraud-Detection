import os
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
