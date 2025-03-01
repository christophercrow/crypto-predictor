"""
Training script for cryptocurrency price prediction model.
"""

import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from crypto_predictor.models.model import build_model
from crypto_predictor.data.data_preprocessing import load_data, clean_data, feature_engineering
from crypto_predictor.utils.config import load_config

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_model(crypto_id, config):
    # Load and preprocess data.
    df = load_data(crypto_id)
    df_clean = clean_data(df)
    df_features = feature_engineering(df_clean)
    
    prices = df_features["price"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    seq_length = config["data"]["sequence_length"]
    X, y = create_sequences(scaled_prices, seq_length)
    X = X.reshape((X.shape[0], seq_length, 1))
    
    lstm_units = config["model"]["lstm_units"]
    model = build_model(seq_length, lstm_units)
    
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    
    config_paths = load_config("config/config.yaml")
    model_dir = config_paths["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{crypto_id}_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="bitcoin", help="Coin ID to train on")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(args.coin, config)
