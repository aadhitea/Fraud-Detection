import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocess_filtering():
    df = pd.DataFrame({
        'type': ['TRANSFER', 'PAYMENT', 'CASH_OUT', 'DEBIT'],
        'amount': [100, 200, 300, 400]
    })
    processed = preprocess_data(df)
    assert len(processed) == 2
    assert set(processed['type_TRANSFER'].unique()).issubset({0, 1})
