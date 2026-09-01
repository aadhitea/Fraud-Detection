import pandas as pd

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
