# src/data_ingestion.py

import requests
import pandas as pd
import datetime
import os

def fetch_crypto_data(crypto_id='bitcoin', vs_currency='usd', days='30'):
    """
    Fetch historical cryptocurrency data from the CoinGecko API.
    
    Args:
        crypto_id (str): The cryptocurrency id (e.g., 'bitcoin').
        vs_currency (str): The currency to compare against (e.g., 'usd').
        days (str): The number of days of data to retrieve.
        
    Returns:
        pd.DataFrame: DataFrame with datetime and price columns.
    """
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()  # Ensure we notice bad responses
    data = response.json()
    
    # The API returns a list of [timestamp, price] pairs
    prices = data.get('prices', [])
    
    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    # Convert timestamp from milliseconds to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'price']]
    
    return df

def save_data(df, crypto_id='bitcoin'):
    """
    Save the fetched DataFrame to a CSV file in the data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        crypto_id (str): The cryptocurrency id to include in the filename.
    """
    # Construct the path relative to the project structure
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f'{crypto_id}_prices.csv')
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    # Example usage: fetch 90 days of Bitcoin data and save it
    df = fetch_crypto_data(days='90')
    save_data(df)
