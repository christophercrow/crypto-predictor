# src/dashboard.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import shap

# Helper to load preprocessed data
def load_data(crypto_id='bitcoin'):
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, f'{crypto_id}_prices_preprocessed.csv')
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return df

# Helper to load our trained model
def load_trained_model():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'lstm_model.h5')
    from tensorflow.keras.losses import MeanSquaredError
    # Load model with custom object for 'mse' and compile=False to bypass compilation issues.
    model = load_model(model_path, custom_objects={"mse": MeanSquaredError()}, compile=False)
    return model

# Create sequences from scaled data and flatten to 2D (samples, seq_length)
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        # data[i:i+seq_length] is (seq_length, 1); flatten to (seq_length,)
        X.append(data[i:i+seq_length].flatten())
    return np.array(X)

# Utility function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def main():
    st.title("Crypto Predictor Dashboard")
    
    # Sidebar selection for cryptocurrency (currently only bitcoin is supported)
    crypto_id = st.sidebar.selectbox("Select Cryptocurrency", ["bitcoin"])
    
    # Load preprocessed data and display a simple line chart of price
    df = load_data(crypto_id)
    st.subheader("Preprocessed Price Data")
    st.line_chart(df['price'])
    
    # Load our trained model
    model = load_trained_model()
    
    # Prepare data for prediction: scale the 'price' column
    prices = df['price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    seq_length = 10
    X = create_sequences(scaled_prices, seq_length)
    
    # Use the last available sequence to predict the next price
    last_seq = X[-1].reshape(1, seq_length)
    
    # Prediction function: reshape 2D input to 3D as required by the model
    def predict_func(x):
        x_reshaped = x.reshape((x.shape[0], seq_length, 1))
        return model.predict(x_reshaped)
    
    prediction = predict_func(last_seq)
    # Inverse-transform the prediction back to original scale
    predicted_price = scaler.inverse_transform(prediction)[0, 0]
    st.write(f"### Predicted Next Price: {predicted_price:.2f}")
    
    # Generate SHAP explanations for the prediction.
    # Use a subset of the data as background for the KernelExplainer.
    background = X[:100] if len(X) > 100 else X
    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape((x.shape[0], seq_length, 1))), background
    )
    
    # Compute SHAP values for the last sequence (our test instance)
    shap_values = explainer.shap_values(last_seq)
    
    # Extract SHAP values for our single test instance.
    # shap_values[0] corresponds to the model output; we extract its first (and only) element.
    shap_vals = shap_values[0][0]
    
    # Adjust the SHAP vector if its length doesn't match the input features length.
    input_length = len(last_seq[0])
    if len(shap_vals) > input_length:
        shap_vals = shap_vals[:input_length]
    elif len(shap_vals) < input_length:
        padding = np.zeros(input_length - len(shap_vals))
        shap_vals = np.concatenate([shap_vals, padding])
    
    # Adjust expected_value if necessary.
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0] if len(expected_value) > 1 else expected_value[0]
    
    # Debug: display lengths to confirm they match.
    st.write(f"Input feature length: {input_length}")
    st.write(f"Adjusted SHAP values length: {len(shap_vals)}")
    
    st.subheader("Model Explanation with SHAP")
    st.write("Below is the SHAP force plot for the current prediction:")
    shap.initjs()
    # Use the adjusted shap_vals instead of shap_values[0][0]
    force_plot = shap.force_plot(expected_value, shap_vals, last_seq[0], matplotlib=False)
    st_shap(force_plot, height=300)

if __name__ == "__main__":
    main()
