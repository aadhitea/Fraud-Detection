import numpy as np
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
