import os
import sys

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest
from src.preprocessing import load_data, clean_data, feature_engineering

@pytest.fixture
def sample_data():
    # Create a small DataFrame to simulate preprocessed data
    df = pd.DataFrame({
        'date': pd.date_range(start='2025-01-01', periods=5, freq='D'),
        'price': [100, 101, 102, 103, 104]
    })
    return df

def test_clean_data(sample_data):
    # Simulate missing values
    sample_data.loc[2, 'price'] = None
    cleaned = clean_data(sample_data)
    # Check if missing value was forward-filled
    assert cleaned.loc[2, 'price'] == 101

def test_feature_engineering(sample_data):
    # Make sure feature engineering runs without error
    cleaned = clean_data(sample_data)
    features = feature_engineering(cleaned)
    # Check if new columns exist
    assert 'price_lag_1' in features.columns
    assert 'rolling_mean_3' in features.columns
