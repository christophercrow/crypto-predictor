#!/usr/bin/env python
"""
Main Pipeline Orchestration Script

This script fetches historical data for all cryptocurrencies specified in the
configuration, then preprocesses the data. It uses modules in the crypto_predictor package.

Usage:
    python -m crypto_predictor.main
"""

import os
from crypto_predictor.data.data_ingestion import fetch_all_coins
from crypto_predictor.data.data_preprocessing import preprocess_all_coins
from crypto_predictor.utils.config import load_config

def main():
    # Load configuration settings.
    config = load_config("config/config.yaml")
    coin_list = config["dashboard"]["available_coins"]
    days = config["data"]["days"]

    print("Starting data ingestion for coins:", coin_list)
    fetch_all_coins(coin_list, days=days)
    print("Data ingestion complete.")

    print("Starting preprocessing...")
    preprocess_all_coins(coin_list)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
