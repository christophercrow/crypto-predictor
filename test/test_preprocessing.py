import pandas as pd
import pytest
from crypto_predictor.data.data_preprocessing import preprocess_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "feature1": [1, 2, None, 4],
        "feature2": [None, 2, 3, 4]
    })

def test_preprocess_data(sample_data):
    preprocess_config = {}
    processed_data = preprocess_data(sample_data, preprocess_config)
    # Ensure no missing values remain
    assert processed_data.isnull().sum().sum() == 0
