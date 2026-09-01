import numpy as np
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
