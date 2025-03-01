import pandas as pd
import logging
from typing import Dict, Any
from pycoingecko import CoinGeckoAPI

logger = logging.getLogger(__name__)

def ingest_data(data_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Ingest historical market data from the CoinGecko API.

    Args:
        data_config: Dictionary with ingestion parameters:
            - coin_id (str): The CoinGecko coin ID (default: 'bitcoin').
            - vs_currency (str): The currency to compare against (default: 'usd').
            - days (str): Number of days to fetch data for. For public API users,
              this should not exceed 365 days (default: '365').

    Returns:
        DataFrame: A pandas DataFrame containing timestamp and price data.
    """
    coin_id = data_config.get("coin_id", "bitcoin")
    vs_currency = data_config.get("vs_currency", "usd")
    days = data_config.get("days", "365")

    # Override 'max' with '365' if necessary, due to public API limitations.
    if days == "max":
        logger.warning("Public API users are limited to querying historical data within the past 365 days. Overriding 'max' to '365'.")
        days = "365"
    
    cg = CoinGeckoAPI()
    logger.info("Fetching market chart data from CoinGecko for coin: %s", coin_id)

    try:
        market_data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    except Exception as e:
        logger.error("Error fetching data from CoinGecko: %s", e)
        raise

    # The API returns data with keys: 'prices', 'market_caps', 'total_volumes'
    # We'll use the 'prices' key, which is a list of [timestamp, price]
    prices = market_data.get("prices", [])
    if not prices:
        logger.error("No price data returned from CoinGecko for coin: %s", coin_id)
        raise ValueError("No price data returned from CoinGecko")

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    # Convert timestamp from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    logger.info("Data ingestion completed with %d rows", len(df))
    return df
