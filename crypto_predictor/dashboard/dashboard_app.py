"""
Streamlit Dashboard for Crypto Price Prediction

This dashboard loads preprocessed data for a selected cryptocurrency,
performs iterative forecasting using a trained LSTM model, and (optionally)
displays SHAP-based model explanations.

Configuration settings are loaded from config/config.yaml.
"""

import os
import sys

# Add the repository root to the Python path so that crypto_predictor package is found.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from crypto_predictor.utils.config import load_config

def load_preprocessed_data(crypto_id):
    """
    Load preprocessed CSV data for a given coin.
    """
    config = load_config("config/config.yaml")
    data_dir = config["paths"]["data_dir"]
    file_path = os.path.join(data_dir, f"{crypto_id}_prices_preprocessed.csv")
    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
    return df

def create_sequences(data, seq_length):
    """
    Convert scaled price data into 2D sequences.
    """
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
    return np.array(X)

def load_trained_model(model_name):
    """
    Load a trained model from the models folder.
    """
    config = load_config("config/config.yaml")
    model_dir = config["paths"]["model_dir"]
    model_path = os.path.join(model_dir, model_name)
    from tensorflow.keras.losses import MeanSquaredError
    model = load_model(model_path, custom_objects={"mse": MeanSquaredError()}, compile=False)
    return model

def model_predict(x, model, seq_length):
    """
    Robust prediction function for use with SHAP.
    
    Ensures that the input x is a NumPy array with a known shape,
    pads or slices it so that each sample has exactly seq_length features,
    reshapes to (n_samples, seq_length, 1), and returns the model's predictions.
    """
    # Convert x to a numpy array with dtype float32.
    x = np.array(x, dtype=np.float32)
    # Ensure x is at least 2D.
    if x.ndim == 1:
        x = x.reshape(1, -1)
    # Force the number of features to be exactly seq_length.
    if x.shape[1] < seq_length:
        pad_width = seq_length - x.shape[1]
        x = np.pad(x, ((0, 0), (0, pad_width)), mode="constant")
    elif x.shape[1] > seq_length:
        x = x[:, :seq_length]
    # Now reshape to (n_samples, seq_length, 1).
    x = x.reshape(x.shape[0], seq_length, 1)
    # Get predictions and force the result to be a numpy array.
    preds = model.predict(x)
    return np.array(preds)

def main():
    st.set_page_config(layout="wide")
    st.title("Crypto Price Prediction Dashboard")
    
    config = load_config("config/config.yaml")
    available_coins = config["dashboard"]["available_coins"]
    seq_length = config["data"]["sequence_length"]

    crypto_id = st.sidebar.selectbox("Select Cryptocurrency", available_coins)
    default_model = f"{crypto_id}_model.h5"
    model_name = st.sidebar.text_input("Model file name:", value=default_model)
    horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=1, max_value=7, value=1)

    try:
        df = load_preprocessed_data(crypto_id)
    except Exception as e:
        st.error(f"Error loading data for {crypto_id}: {e}")
        return

    st.subheader(f"Historical Price Data for {crypto_id.capitalize()}")
    st.line_chart(df["price"])

    prices = df["price"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    X = create_sequences(scaled_prices, seq_length)

    try:
        model = load_trained_model(model_name)
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")
        return

    if len(X) == 0:
        st.error("Not enough data to create prediction sequences.")
        return

    last_seq = X[-1].reshape(1, seq_length)
    predictions = []
    current_seq = last_seq.copy()
    for _ in range(horizon):
        pred_scaled = model.predict(current_seq).flatten()
        next_seq = np.append(current_seq.flatten()[1:], pred_scaled)
        current_seq = next_seq.reshape(1, seq_length)
        predictions.append(pred_scaled[0])
    predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    st.write(f"### {horizon}-Day Forecast for {crypto_id.capitalize()}")
    st.write("Predicted Prices:", predictions_inv)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["price"], label="Historical Price")
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=horizon+1, freq="D")[1:]
    ax.plot(future_dates, predictions_inv, label="Predicted Price", marker="o")
    ax.legend()
    st.pyplot(fig)

    if st.checkbox("Show SHAP Explanation"):
        st.subheader("Model Explanation with SHAP")
        background = np.array(X[:100]) if len(X) > 100 else np.array(X)
        explainer = shap.KernelExplainer(lambda x: model_predict(x, model, seq_length), background)
        try:
            shap_values = explainer.shap_values(last_seq)
        except Exception as e:
            st.error(f"Error computing SHAP values: {e}")
            return
        shap_vals = shap_values[0][0]
        if len(shap_vals) > seq_length:
            shap_vals = shap_vals[:seq_length]
        elif len(shap_vals) < seq_length:
            shap_vals = np.concatenate([shap_vals, np.zeros(seq_length - len(shap_vals))])
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]
        shap.initjs()
        force_plot = shap.force_plot(expected_value, shap_vals, last_seq[0], matplotlib=False)
        import streamlit.components.v1 as components
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        components.html(shap_html, height=300)

if __name__ == "__main__":
    main()
