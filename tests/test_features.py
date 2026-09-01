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
    assert feats['origin_balance_error'].iloc[0] == 0.0
