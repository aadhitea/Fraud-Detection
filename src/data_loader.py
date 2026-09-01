import os
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
