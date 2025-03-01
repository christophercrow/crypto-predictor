# src/dashboard.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import shap

def load_preprocessed_data(crypto_id='bitcoin'):
    """
    Load the preprocessed data for a given coin.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, f'{crypto_id}_prices_preprocessed.csv')
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return df

def create_sequences(data, seq_length):
    """
    Convert scaled price data into sequences for LSTM input.
    """
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].flatten())
    return np.array(X)

def load_trained_model(model_name='lstm_model.h5'):
    """
    Load a trained LSTM model.
    """
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, model_name)
    from tensorflow.keras.losses import MeanSquaredError
    model = load_model(model_path, custom_objects={"mse": MeanSquaredError()}, compile=False)
    return model

def main():
    st.title("Crypto Predictor Dashboard")
    
    # Sidebar for coin selection
    coin_list = ["bitcoin", "ethereum", "cardano"]
    crypto_id = st.sidebar.selectbox("Select Cryptocurrency", coin_list)
    
    # Sidebar for model name (in case you have multiple saved models)
    model_name = st.sidebar.text_input("Model file name:", value="lstm_model.h5")
    
    # Load preprocessed data for the selected coin
    df = load_preprocessed_data(crypto_id)
    st.subheader(f"Preprocessed Price Data: {crypto_id}")
    st.line_chart(df['price'])
    
    # Prepare data for prediction
    prices = df['price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    seq_length = 10
    X = create_sequences(scaled_prices, seq_length)
    
    # Load the trained LSTM model
    model = load_trained_model(model_name)
    
    # Use the last sequence to predict the next price
    if len(X) > 0:
        last_seq = X[-1].reshape(1, seq_length)
        
        # Reshape for LSTM input
        def predict_func(x):
            x_reshaped = x.reshape((x.shape[0], seq_length, 1))
            return model.predict(x_reshaped)
        
        prediction_scaled = predict_func(last_seq)
        predicted_price = scaler.inverse_transform(prediction_scaled)[0, 0]
        st.write(f"### Predicted Next Price: {predicted_price:.2f}")
        
        # SHAP Explanation (Optional)
        if st.checkbox("Show SHAP Explanation"):
            st.subheader("SHAP Force Plot")
            background = X[:100] if len(X) > 100 else X
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x.reshape((x.shape[0], seq_length, 1))), background
            )
            
            shap_values = explainer.shap_values(last_seq)
            shap_vals = shap_values[0][0]
            
            # Adjust lengths if needed
            input_length = seq_length
            if len(shap_vals) > input_length:
                shap_vals = shap_vals[:input_length]
            elif len(shap_vals) < input_length:
                shap_vals = np.concatenate([shap_vals, np.zeros(input_length - len(shap_vals))])
            
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[0]
            
            shap.initjs()
            force_plot = shap.force_plot(expected_value, shap_vals, last_seq[0], matplotlib=False)
            
            import streamlit.components.v1 as components
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(shap_html, height=300)
    else:
        st.write("Not enough data to create sequences for LSTM prediction.")

if __name__ == "__main__":
    main()
