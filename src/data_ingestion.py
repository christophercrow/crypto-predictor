# src/data_ingestion.py

import requests
import pandas as pd
import datetime
import os

def fetch_crypto_data(crypto_id='bitcoin', vs_currency='usd', days='30'):
    """
    Fetch historical cryptocurrency data from the CoinGecko API.
    
    Args:
        crypto_id (str): The cryptocurrency ID (e.g., 'bitcoin', 'ethereum').
        vs_currency (str): The currency to compare against (e.g., 'usd').
        days (str): Number of days of data to retrieve (e.g., '30', 'max').
        
    Returns:
        pd.DataFrame: DataFrame with columns [date, price].
    """
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an exception for non-2xx responses
    data = response.json()
    
    prices = data.get('prices', [])
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'price']]
    
    return df

def save_data(df, crypto_id='bitcoin'):
    """
    Save the fetched DataFrame to a CSV file in the data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        crypto_id (str): The cryptocurrency ID to include in the filename.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f'{crypto_id}_prices.csv')
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def fetch_all_coins(coin_list, vs_currency='usd', days='30'):
    """
    Fetch data for all coins in coin_list and save each to a separate CSV.
    
    Args:
        coin_list (list): List of coin IDs (e.g., ['bitcoin', 'ethereum']).
        vs_currency (str): The currency to compare against.
        days (str): Number of days of data to retrieve.
    """
    for coin in coin_list:
        print(f"Fetching data for {coin}...")
        df = fetch_crypto_data(crypto_id=coin, vs_currency=vs_currency, days=days)
        save_data(df, crypto_id=coin)

if __name__ == "__main__":
    # Example usage: fetch data for multiple coins
    coins = ["bitcoin", "ethereum", "cardano"]
    fetch_all_coins(coins, days='90')
