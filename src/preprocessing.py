# src/preprocessing.py

import pandas as pd
import os

def load_data(crypto_id='bitcoin'):
    """
    Load raw cryptocurrency data from the CSV file.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, f'{crypto_id}_prices.csv')
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df

def clean_data(df):
    """
    Clean the dataset by removing duplicates and filling missing values.
    """
    df = df.drop_duplicates()
    df = df.fillna(method='ffill')
    return df

def feature_engineering(df):
    """
    Perform feature engineering to create additional informative features.
    """
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    # Example: Create lag features
    for lag in range(1, 4):
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Example: rolling mean and std
    df['rolling_mean_3'] = df['price'].rolling(window=3).mean()
    df['rolling_std_3'] = df['price'].rolling(window=3).std()
    
    df.dropna(inplace=True)
    return df

def preprocess_coin(crypto_id='bitcoin'):
    """
    Preprocess data for a single coin and save to a new CSV.
    """
    df = load_data(crypto_id)
    df_clean = clean_data(df)
    df_features = feature_engineering(df_clean)
    
    # Save preprocessed data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    out_path = os.path.join(data_dir, f'{crypto_id}_prices_preprocessed.csv')
    df_features.to_csv(out_path)
    print(f"Preprocessed data saved to {out_path}")

def preprocess_all_coins(coin_list):
    """
    Preprocess data for all coins in coin_list.
    """
    for coin in coin_list:
        print(f"Preprocessing data for {coin}...")
        preprocess_coin(coin)

if __name__ == "__main__":
    coins = ["bitcoin", "ethereum", "cardano"]
    preprocess_all_coins(coins)
