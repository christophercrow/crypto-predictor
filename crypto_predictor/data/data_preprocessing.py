import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame, preprocess_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess the raw data for training.
    Fills missing values and ensures data is sorted by timestamp.
    
    Args:
        data: Raw data as a DataFrame.
        preprocess_config: Dictionary with preprocessing parameters.
        
    Returns:
        Processed DataFrame.
    """
    logger.info("Starting data preprocessing")
    data = data.copy()
    data.sort_values("timestamp", inplace=True)
    # Using ffill() instead of fillna(method="ffill") per FutureWarning
    data.ffill(inplace=True)
    logger.info("Data preprocessing completed")
    return data

def prepare_sequences(data: pd.DataFrame, window_size: int = 60) -> Dict[str, Any]:
    """
    Prepare training sequences for LSTM model using a sliding window approach.
    
    Args:
        data: DataFrame with a 'price' column.
        window_size: Number of time steps to include in each input sequence.
        
    Returns:
        Dictionary with keys 'X_train' and 'y_train'.
    """
    prices = data["price"].values
    X, y = [], []
    for i in range(len(prices) - window_size):
        X.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    X = np.array(X)
    y = np.array(y)
    # Reshape X for LSTM input: (samples, timesteps, features)
    X = X.reshape(-1, window_size, 1)
    return {"X_train": X, "y_train": y}
