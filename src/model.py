# src/model.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def load_data(crypto_id='bitcoin'):
    """
    Load the preprocessed cryptocurrency price data.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    file_path = os.path.join(data_dir, f'{crypto_id}_prices_preprocessed.csv')
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    return df

def create_sequences(data, seq_length):
    """
    Create input sequences and corresponding targets for training.
    
    Args:
        data (np.array): Array of scaled price values.
        seq_length (int): Number of time steps in each input sequence.
    
    Returns:
        X (np.array): Input sequences.
        y (np.array): Corresponding target values.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def build_model(input_shape):
    """
    Build and compile an LSTM model.
    
    Args:
        input_shape (tuple): Shape of the input data (time steps, features).
    
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Load preprocessed data
    df = load_data()
    
    # We'll use only the 'price' column for forecasting
    prices = df['price'].values.reshape(-1, 1)
    
    # Scale the data to [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Define the sequence length (number of time steps)
    seq_length = 10
    
    # Create sequences from the scaled data
    X, y = create_sequences(scaled_prices, seq_length)
    
    # Reshape X to be [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split the data into training and testing sets (e.g., 80/20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build the LSTM model
    model = build_model((seq_length, 1))
    model.summary()  # Print model architecture
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=16,
        callbacks=[early_stop]
    )
    
    # Evaluate the model on test data
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    
    # Generate predictions on the test set
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(predictions_inv, label='Predicted Price')
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.show()
    
    # Save the trained model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lstm_model.h5')
    model.save(model_path)
    print("Model saved to", model_path)

if __name__ == "__main__":
    main()
