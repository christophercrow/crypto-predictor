import os
import sys
import pandas as pd
import pytest

# Add repository root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from crypto_predictor.data.data_preprocessing import clean_data, feature_engineering

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "date": pd.date_range(start="2025-01-01", periods=5, freq="D"),
        "price": [100, 101, 102, 103, 104]
    })
    return df

def test_clean_data(sample_data):
    sample_data.loc[2, "price"] = None
    cleaned = clean_data(sample_data)
    assert cleaned.loc[2, "price"] == 101

def test_feature_engineering(sample_data):
    cleaned = clean_data(sample_data)
    features = feature_engineering(cleaned)
    assert "price_lag_1" in features.columns
    assert "rolling_mean_3" in features.columns
