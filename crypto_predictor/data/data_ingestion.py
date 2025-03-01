"""
Module to fetch historical cryptocurrency data from CoinGecko.
"""

import requests
import pandas as pd
import os
from crypto_predictor.utils.config import load_config

def fetch_crypto_data(crypto_id, vs_currency="usd", days="30"):
    """
    Fetch historical data from CoinGecko for a specific coin.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["date", "price"]]
    return df

def save_data(df, crypto_id):
    """
    Save the fetched DataFrame as CSV.
    """
    config = load_config("config/config.yaml")
    data_dir = config["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{crypto_id}_prices.csv")
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def fetch_all_coins(coin_list, vs_currency="usd", days="30"):
    """
    Fetch data for all coins in coin_list.
    """
    for coin in coin_list:
        print(f"Fetching data for {coin}...")
        df = fetch_crypto_data(coin, vs_currency, days)
        save_data(df, coin)

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    coin_list = config["dashboard"]["available_coins"]
    fetch_all_coins(coin_list, days=config["data"]["days"])
