# Crypto Predictor

Crypto Predictor is an end-to-end machine learning project for forecasting cryptocurrency prices using LSTM models with SHAP-based interpretability. This repository is organized as an installable package, making imports and maintenance easier.

## Repository Structure

crypto-predictor/ ├── crypto_predictor/ # Python package │ ├── init.py │ ├── main.py # Main pipeline orchestration script │ ├── data/ # Data ingestion and preprocessing modules │ │ ├── init.py │ │ ├── coin_list.py │ │ ├── data_ingestion.py │ │ └── data_preprocessing.py │ ├── models/ # Model definition and training │ │ ├── init.py │ │ ├── model.py │ │ └── train.py │ ├── utils/ # Utility functions (e.g., config loader) │ │ ├── init.py │ │ └── config.py │ └── dashboard/ # Streamlit dashboard application │ ├── init.py │ └── dashboard_app.py ├── config/ │ └── config.yaml # Configuration file ├── tests/ │ └── test_preprocessing.py # Unit tests for data preprocessing ├── .github/ │ └── workflows/ │ └── ci.yml # GitHub Actions workflow for CI ├── Dockerfile # Dockerfile to build container ├── README.md # This file └── setup.py # Package installation script


## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/christophercrow/crypto-predictor.git
   cd crypto-predictor

    Create a Virtual Environment & Install Dependencies:

    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install -e .

    Configuration: Adjust settings in config/config.yaml if needed.

Usage

    Data Ingestion & Preprocessing:

python -m crypto_predictor.main

This will fetch data for all coins in the configuration and preprocess it.

Training a Model (example for Bitcoin):

python -m crypto_predictor.models.train --coin bitcoin --config config/config.yaml

The trained model is saved as specified in the config.

Launch the Dashboard:

streamlit run crypto_predictor/dashboard/dashboard_app.py

Run Tests:

    pytest --maxfail=1 --disable-warnings -q

Docker

To build and run the Docker container:

docker build -t crypto-predictor .
docker run -p 8501:8501 crypto-predictor

License

MIT License