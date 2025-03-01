"""
Module for building the LSTM model.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(seq_length, lstm_units=50):
    """
    Build an LSTM model for price prediction.
    """
    model = Sequential([
        tf.keras.layers.Input(shape=(seq_length, 1)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.2),
        LSTM(lstm_units),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

if __name__ == "__main__":
    model = build_model(seq_length=10)
    model.summary()
