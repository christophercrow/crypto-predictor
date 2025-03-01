"""
Module to fetch a dynamic list of supported coins from CoinGecko.
"""

import requests

def get_supported_coins():
    """
    Fetch the list of coins from CoinGecko API.
    
    Returns:
        List of dictionaries with coin 'id', 'symbol', and 'name'.
    """
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url)
    response.raise_for_status()
    coins = response.json()
    return coins

if __name__ == "__main__":
    coins = get_supported_coins()
    for coin in coins[:10]:
        print(coin)
