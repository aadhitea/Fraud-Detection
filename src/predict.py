import numpy as np
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
