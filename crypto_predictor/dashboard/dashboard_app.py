import streamlit as st
import logging
import pandas as pd
import numpy as np
import altair as alt
from tensorflow.keras.models import load_model
from crypto_predictor.data.data_ingestion import ingest_data
from crypto_predictor.data.data_preprocessing import preprocess_data, prepare_sequences
from crypto_predictor.utils.config import load_config
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X, y):
    """
    Evaluate the model on the given test sequences and compute additional metrics.
    """
    loss = model.evaluate(X, y, verbose=0)
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return loss, mae, rmse, predictions

def plot_historical_data(data: pd.DataFrame, predictions: np.ndarray = None, window_size: int = 60):
    """
    Plot historical price data. If predictions are provided, overlay the prediction for the last window.
    """
    chart = alt.Chart(data).mark_line().encode(
        x='timestamp:T',
        y='price:Q'
    ).properties(
        title="Historical Price Data"
    )
    st.altair_chart(chart, use_container_width=True)

    if predictions is not None:
        # Create a DataFrame for the prediction, using the last timestamp as a reference
        last_time = data['timestamp'].iloc[-1]
        pred_time = last_time + pd.Timedelta(minutes=window_size)  # assuming minute frequency; adjust as needed
        pred_df = pd.DataFrame({
            'timestamp': [pred_time],
            'price': [predictions[0][0]]
        })
        pred_chart = alt.Chart(pred_df).mark_point(color='red', size=100).encode(
            x='timestamp:T',
            y='price:Q'
        ).properties(
            title="Latest Prediction"
        )
        st.altair_chart(pred_chart, use_container_width=True)

def run_dashboard():
    st.title("Crypto Predictor Dashboard")
    
    # Sidebar for interactive settings
    st.sidebar.header("Settings")
    coin_id = st.sidebar.text_input("CoinGecko Coin ID", value="bitcoin")
    vs_currency = st.sidebar.text_input("Vs Currency", value="usd")
    days = st.sidebar.text_input("Days of Data (max 365)", value="365")
    window_size = st.sidebar.number_input("Window Size", min_value=10, max_value=200, value=60, step=10)
    
    # Load base config and override with sidebar inputs
    config = load_config("config/config.yaml")
    config["data"]["coin_id"] = coin_id
    config["data"]["vs_currency"] = vs_currency
    config["data"]["days"] = days
    config["preprocessing"]["window_size"] = window_size
    save_path = config.get("training", {}).get("save_path", "models/saved_model.keras")
    if not (save_path.endswith(".keras") or save_path.endswith(".h5")):
        save_path += ".keras"
    config["training"]["save_path"] = save_path

    st.subheader("Data Ingestion & Preprocessing")
    try:
        raw_data = ingest_data(config.get("data", {}))
        st.write("Raw data preview:", raw_data.head())
        processed_data = preprocess_data(raw_data, config.get("preprocessing", {}))
        st.write("Processed data preview:", processed_data.head())
        sequences = prepare_sequences(processed_data, window_size)
        X, y = sequences["X_train"], sequences["y_train"]
        st.write(f"Prepared {X.shape[0]} sequences for model evaluation.")
    except Exception as e:
        st.error(f"Error during data ingestion or preprocessing: {e}")
        logger.error("Dashboard data error: %s", e)
        return

    st.subheader("Model Evaluation")
    try:
        model = load_model(save_path)
        loss, mae, rmse, predictions = evaluate_model(model, X, y)
        st.write(f"Evaluation Loss: {loss:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"Root Mean Squared Error: {rmse:.4f}")
    except Exception as e:
        st.error(f"Error during model evaluation: {e}")
        logger.error("Dashboard evaluation error: %s", e)
        return

    st.subheader("Visualization")
    try:
        plot_historical_data(processed_data, predictions=predictions[-1:], window_size=window_size)
    except Exception as e:
        st.error(f"Error during visualization: {e}")
        logger.error("Dashboard visualization error: %s", e)

    st.subheader("Latest Prediction")
    try:
        if X.shape[0] > 0:
            last_sequence = X[-1].reshape(1, X.shape[1], 1)
            latest_pred = model.predict(last_sequence)
            st.write("Latest Price Prediction:", latest_pred[0][0])
        else:
            st.write("Not enough data for predictions.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        logger.error("Dashboard prediction error: %s", e)

if __name__ == "__main__":
    run_dashboard()
