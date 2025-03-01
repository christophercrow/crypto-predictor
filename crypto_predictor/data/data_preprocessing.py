"""
Module for data cleaning and feature engineering.
"""

import pandas as pd
import os
from crypto_predictor.utils.config import load_config

def load_data(crypto_id):
    """
    Load raw CSV data for a coin.
    """
    config = load_config("config/config.yaml")
    data_dir = config["paths"]["data_dir"]
    file_path = os.path.join(data_dir, f"{crypto_id}_prices.csv")
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

def clean_data(df):
    """
    Clean the dataset by removing duplicates and forward-filling missing values.
    """
    df = df.drop_duplicates()
    df = df.ffill()
    return df

def feature_engineering(df):
    """
    Add lag features and rolling statistics.
    """
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    for lag in range(1, 4):
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
    df["rolling_mean_3"] = df["price"].rolling(window=3).mean()
    df["rolling_std_3"] = df["price"].rolling(window=3).std()
    df.dropna(inplace=True)
    return df

def preprocess_coin(crypto_id):
    """
    Preprocess data for a coin and save to CSV.
    """
    df = load_data(crypto_id)
    df_clean = clean_data(df)
    df_features = feature_engineering(df_clean)
    config = load_config("config/config.yaml")
    data_dir = config["paths"]["data_dir"]
    file_path = os.path.join(data_dir, f"{crypto_id}_prices_preprocessed.csv")
    df_features.to_csv(file_path)
    print(f"Preprocessed data saved to {file_path}")

def preprocess_all_coins(coin_list):
    """
    Preprocess data for each coin in coin_list.
    """
    for coin in coin_list:
        print(f"Preprocessing data for {coin}...")
        preprocess_coin(coin)

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    coin_list = config["dashboard"]["available_coins"]
    preprocess_all_coins(coin_list)
